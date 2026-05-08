"""Rejection-sampling SFT 轨迹生成（Pure ReAct + 单 retrieval 工具版）。

流程：
1. 教师模型（DashScope Qwen，通过 langchain ChatTongyi 的 bind_tools 走原生 function calling）
   在新 system prompt 下，对每条 query 采样 K 条 trajectory
2. 每条 trajectory 用 reward_rag.compute_score 打分
3. 只保留 reward >= 阈值（默认 1.0，即任务成功）的 trajectory
4. 输出 JSONL，messages 字段保留完整多轮 chat（system/user/assistant+tool_calls/tool）

为什么这样做：
- 比纯 offline SFT 更接近 on-policy（教师"挑出自己分布上 reward 高的样本"）
- 比真 GKD / MiniLLM 简单 10 倍：API 调用一次性离线做完，不进训练 loop
- 教师产物的 tool 调用就是 OpenAI function calling 标准 JSON，与 student 在
  VeRL multi-turn rollout 阶段的输出格式一致，避免 SFT/GRPO 格式漂移

用法：
    export QWEN_API_KEY=...
    python scripts/build_sft_trajectories_react.py \
        --queries_file data/eval/queries.txt \
        --output_file data/verl_trajs/trajs_react_filtered.jsonl \
        --k_per_query 6 \
        --reward_threshold 1.0 \
        --max_workers 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# 让 src.* 与 scripts.* 在子进程里能 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.chat_models import ChatTongyi  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool  # noqa: E402

# 复用 src 里的向量库 & 重排逻辑（注意：教师采样不依赖 verl_tools，因为这是离线脚本）
from src.tools.retrieval import retrieval_augment as _lc_retrieval  # noqa: E402

# 复用 reward 函数：直接用我们 multi-turn 版的 compute_score
from scripts.reward_rag import compute_score  # noqa: E402

logger = logging.getLogger(__name__)

# 教师 system prompt 必须和 build_grpo_prompt_parquet.py 的 SYSTEM_PROMPT 保持一致，
# 否则 SFT 学到的格式与 GRPO 期望的格式不一致。这里直接复用。
from scripts.build_grpo_prompt_parquet import SYSTEM_PROMPT  # noqa: E402


# ---------------------------------------------------------------------------
# 工具定义：langchain @tool，给教师 bind_tools 用
# ---------------------------------------------------------------------------


@tool
def retrieval_augment(query: str, keyword: str = "") -> str:
    """Search the school regulation knowledge base. Returns top-5 relevant document chunks.

    Args:
        query: The search query (you may rewrite the user's question for better retrieval).
        keyword: A short keyword for keyword-aware reranking. Pass empty string if none.
    """
    # 直接调 src 里那个 langchain @tool 的底层逻辑（它本身也是 @tool）
    return _lc_retrieval.invoke({"query": query, "keyword": keyword or ""})


# ---------------------------------------------------------------------------
# 教师 ReAct 循环
# ---------------------------------------------------------------------------


def lc_messages_to_dicts(messages: list) -> list[dict]:
    """langchain Message → JSON 可序列化 dict（OpenAI chat 格式）。"""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            d: dict = {"role": "assistant", "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                # langchain tool_calls: [{"id": ..., "name": ..., "args": {...}}]
                # 转成 OpenAI function calling 标准
                d["tool_calls"] = [
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "content": m.content if isinstance(m.content, str) else str(m.content),
                    "tool_call_id": m.tool_call_id,
                }
            )
    return out


def messages_to_solution_str(messages_dict: list[dict]) -> str:
    """渲染成 reward_rag 能识别的 chat-template 字符串（Qwen3 风格）。

    格式：<|im_start|>{role}\n{content}<|im_end|>
    assistant 含 tool_calls 时，把 <tool_call>JSON</tool_call> 拼到 content 里
    （reward_rag 的 _TOOL_CALL_TAG 正则会数 <tool_call> 标签）。
    """
    parts: list[str] = []
    for m in messages_dict:
        role = m["role"]
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            tc_strs = []
            for tc in m["tool_calls"]:
                args_str = tc["function"]["arguments"]
                if not isinstance(args_str, str):
                    args_str = json.dumps(args_str, ensure_ascii=False)
                tc_strs.append(
                    f'<tool_call>\n{{"name": "{tc["function"]["name"]}", '
                    f'"arguments": {args_str}}}\n</tool_call>'
                )
            full = (content + "\n" if content else "") + "\n".join(tc_strs)
            parts.append(f"<|im_start|>assistant\n{full}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def run_teacher_react(
    llm_with_tools,
    query: str,
    max_turns: int = 5,
) -> tuple[list, str, int]:
    """跑一次教师 Pure ReAct 循环，返回 (langchain messages, final_answer, n_tool_calls)。"""
    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    n_tool_calls = 0
    final_answer = ""

    for turn in range(max_turns):
        try:
            response: AIMessage = llm_with_tools.invoke(messages)
        except Exception as e:
            logger.warning(f"Teacher invoke failed at turn {turn}: {e}")
            break

        messages.append(response)

        if not getattr(response, "tool_calls", None):
            # 没有 tool call → 视为最终答案
            final_answer = response.content or ""
            break

        # 有 tool call：执行
        for tc in response.tool_calls:
            n_tool_calls += 1
            name = tc["name"]
            args = tc.get("args", {}) or {}
            try:
                if name == "retrieval_augment":
                    tool_result = retrieval_augment.invoke(args)
                else:
                    tool_result = f"Error: unknown tool {name!r}"
            except Exception as e:
                tool_result = f"Error executing {name}: {e}"
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc.get("id") or f"call_{turn}_{n_tool_calls}",
                )
            )
    else:
        # 循环 break 走 else：没找到 final answer，把最后一条 assistant 当作答案
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                final_answer = m.content
                break

    return messages, final_answer, n_tool_calls


# ---------------------------------------------------------------------------
# 单 query 的 K 次采样 + reward 过滤
# ---------------------------------------------------------------------------


def sample_and_filter_one_query(
    query: str,
    ground_truth: Any,
    key_evidence: list,
    keywords: list,
    teacher_model_name: str,
    api_key: str,
    k_per_query: int,
    max_turns: int,
    reward_threshold: float,
    temperature: float,
) -> list[dict]:
    """对单条 query 采样 K 次，返回 reward >= threshold 的 trajectory dicts。"""
    llm = ChatTongyi(
        model=teacher_model_name,
        dashscope_api_key=api_key,
        temperature=temperature,
    )
    llm_with_tools = llm.bind_tools([retrieval_augment])

    gt_dict = {
        "answer": ground_truth if isinstance(ground_truth, str) else "",
        "keywords": keywords or [],
    }
    extra_info = {
        "key_evidence": key_evidence or [],
        "raw_query": query,
    }

    accepted: list[dict] = []
    for k in range(k_per_query):
        try:
            lc_msgs, final_answer, n_tool_calls = run_teacher_react(
                llm_with_tools, query, max_turns=max_turns
            )
        except Exception as e:
            logger.warning(f"[{query[:40]}...] sample {k} failed: {e}")
            continue

        msgs_dict = lc_messages_to_dicts(lc_msgs)
        # 算 reward
        sol_str = messages_to_solution_str(msgs_dict)
        try:
            reward = compute_score(
                data_source="agentic_rag",
                solution_str=sol_str,
                ground_truth=gt_dict,
                extra_info=extra_info,
            )
        except Exception as e:
            logger.warning(f"[{query[:40]}...] reward failed: {e}")
            continue

        record = {
            "query": query,
            "ground_truth": ground_truth,
            "messages": msgs_dict,
            "final_answer": final_answer,
            "n_tool_calls": n_tool_calls,
            "reward": reward,
            "task_success": reward >= 1.0,
            "sample_idx": k,
            "key_evidence": key_evidence or [],
            "keywords": keywords or [],
        }

        if reward >= reward_threshold:
            accepted.append(record)

    return accepted


# ---------------------------------------------------------------------------
# 输入加载 / 断点续跑
# ---------------------------------------------------------------------------


def load_input_rows(queries_file: str, trajs_file: Optional[str]) -> list[dict]:
    """读输入 query。若给了 trajs_file（旧 SFT 轨迹）则带 ground_truth + key_evidence。"""
    rows: list[dict] = []
    if trajs_file and Path(trajs_file).exists():
        with open(trajs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (obj.get("query") or "").strip()
                if not q:
                    continue
                gt = obj.get("final_answer") or ""
                if not gt and "messages" in obj:
                    for m in reversed(obj["messages"]):
                        if (m.get("role") or "").lower() == "assistant":
                            gt = (m.get("content") or "").strip()
                            break
                rows.append(
                    {
                        "query": q,
                        "ground_truth": gt,
                        "key_evidence": obj.get("key_evidence") or [],
                        "keywords": obj.get("keywords") or [],
                    }
                )
    elif queries_file and Path(queries_file).exists():
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    rows.append(
                        {"query": q, "ground_truth": "", "key_evidence": [], "keywords": []}
                    )
    else:
        raise FileNotFoundError("--queries_file 和 --trajs_file 都没找到。")
    return rows


def load_finished_queries(output_path: str) -> set:
    p = Path(output_path)
    if not p.exists():
        return set()
    finished = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get("query")
                if isinstance(q, str):
                    finished.add(q)
            except Exception:
                continue
    return finished


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejection-sampling SFT 轨迹生成")
    parser.add_argument("--queries_file", type=str, default="./data/eval/queries.txt")
    parser.add_argument(
        "--trajs_file",
        type=str,
        default="",
        help="可选：旧的 trajs.jsonl，用于带 ground_truth + key_evidence",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/verl_trajs/trajs_react_filtered.jsonl",
    )
    parser.add_argument("--teacher_model", type=str, default="qwen-plus")
    parser.add_argument("--k_per_query", type=int, default=6, help="每条 query 采样几次")
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=1.0,
        help="保留 reward >= 阈值的 trajectory（1.0 = 任务成功）",
    )
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--max_workers", type=int, default=5, help="并发 worker 数（受 DashScope 速率限制）"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="只跑前 N 条 query（试跑时用）"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 QWEN_API_KEY。")

    rows = load_input_rows(args.queries_file, args.trajs_file or None)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没读到任何 query。")

    finished = load_finished_queries(args.output_file)
    pending = [r for r in rows if r["query"] not in finished]
    logger.info(
        f"总 {len(rows)} 条 query，已完成 {len(finished)}，待处理 {len(pending)}，"
        f"K={args.k_per_query}，threshold={args.reward_threshold}"
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    fw = open(args.output_file, "a", encoding="utf-8")

    total_accepted = 0
    total_processed = 0
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_row = {
                pool.submit(
                    sample_and_filter_one_query,
                    row["query"],
                    row["ground_truth"],
                    row["key_evidence"],
                    row["keywords"],
                    args.teacher_model,
                    api_key,
                    args.k_per_query,
                    args.max_turns,
                    args.reward_threshold,
                    args.temperature,
                ): row
                for row in pending
            }
            for fut in as_completed(future_to_row):
                row = future_to_row[fut]
                try:
                    accepted = fut.result()
                except Exception as e:
                    logger.warning(f"query 失败: {row['query'][:50]}... -> {e}")
                    accepted = []

                total_processed += 1
                for rec in accepted:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fw.flush()
                total_accepted += len(accepted)

                if total_processed % 10 == 0 or total_processed == len(pending):
                    elapsed = time.time() - t0
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[{total_processed}/{len(pending)}] 已接受 {total_accepted} "
                        f"({rate:.1f} q/s, accept rate={total_accepted/(total_processed*args.k_per_query):.1%})"
                    )
    finally:
        fw.close()

    logger.info(
        f"完成。处理 {total_processed} query，接受 {total_accepted} 条 trajectory，"
        f"输出 -> {args.output_file}"
    )


if __name__ == "__main__":
    main()
