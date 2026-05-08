"""v1 (LangChain 3 工具) vs v2 (Pure ReAct 1 工具) 端到端对比脚本。

目的：在相同 query 集合上跑两套架构，输出可量化的 4 个指标对比表，
用于简历量化数据 / 面试讲故事。

两边都用同一个主 LLM（默认 qwen-plus）以保证对比公平：
- v1：LangChain `RagAgent`，3 个工具（expand_and_keyword / retrieval / summary），
  expand 和 summary 内部还会调一次 Qwen API
- v2：Pure ReAct loop，仅 retrieval 一个工具（本地 Chroma，0 API 调用），
  query 改写、关键词、文档精读都内化到模型 chain-of-thought

⚠️ v2 这里用的是**未经训练**的 qwen-plus + Pure ReAct prompt，所以这个对比
   反映的是**架构层面的改进**（工具数 / API 依赖 / 延迟）；训练带来的
   推理质量提升（21.2% 推理效率等）需要额外做 base-vs-trained 对比。

用法：
    export QWEN_API_KEY=...
    python scripts/v1_vs_v2_compare.py \
        --queries data/eval/queries.txt \
        --trajs data/verl_trajs/trajs.jsonl \
        --output_dir data/eval/v1_vs_v2 \
        --limit 30

输出：
    data/eval/v1_vs_v2/v1_results.jsonl     # 每条 query 的 v1 trajectory
    data/eval/v1_vs_v2/v2_results.jsonl     # 每条 query 的 v2 trajectory
    data/eval/v1_vs_v2/summary.md           # 聚合对比表（直接抠数字进简历）
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# 让 src.* / scripts.* 可 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.chat_models import ChatTongyi  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool  # noqa: E402

from src.agents.rag_agent import RagAgent  # noqa: E402
from src.tools.retrieval import retrieval_augment as _src_retrieval  # noqa: E402
from scripts.build_grpo_prompt_parquet import SYSTEM_PROMPT  # noqa: E402


# ===========================================================================
# v2 用的 retrieval tool（langchain @tool，bind_tools 给 LLM）
# 工具描述要尽量贴近 GRPO 训练时的 schema，否则模型行为有偏
# ===========================================================================


@tool
def retrieval_augment(query: str, keyword: str = "") -> str:
    """Search the school regulation knowledge base. Returns top-5 relevant chunks.

    Args:
        query: The search query (you may rewrite the user's question for better retrieval).
        keyword: A short keyword for keyword-aware reranking. Pass empty string if none.
    """
    return _src_retrieval.invoke({"query": query, "keyword": keyword or ""})


# ===========================================================================
# 简单任务匹配（避免 reward_rag.compute_score 的 chat-template marker 依赖）
# ===========================================================================


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def task_match(final_answer: str, ground_truth: str, keywords: list) -> bool:
    """final_answer 是否覆盖 ground_truth / keywords 一半以上。"""
    if not final_answer:
        return False
    norm = _norm(final_answer)
    if ground_truth:
        if _norm(ground_truth) in norm:
            return True
    if keywords:
        hits = sum(1 for k in keywords if k and _norm(str(k)) in norm)
        if hits >= max(1, len(keywords) * 0.5):
            return True
    return False


# ===========================================================================
# v1 runner
# ===========================================================================


def _msg_brief(m) -> dict:
    cls_name = type(m).__name__
    content = getattr(m, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    snippet = content[:200] + ("..." if len(content) > 200 else "")
    tcs = getattr(m, "tool_calls", None) or []
    return {
        "type": cls_name,
        "content_snippet": snippet,
        "tool_calls": [
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
            for tc in tcs
        ],
    }


def run_v1(query: str, llm) -> dict:
    """跑 v1 LangChain RagAgent。"""
    t0 = time.time()
    error = None
    n_tool_calls = 0
    n_assistant_turns = 0
    by_name = {"expand_and_keyword": 0, "retrieval_augment": 0, "summary_related_doc": 0}
    final_answer = ""
    msgs_dump: list = []
    raw_messages = []

    try:
        agent = RagAgent(llm=llm)
        # 直接调底层 graph，能拿到完整 messages
        result = agent._agent.invoke({"messages": [("user", query)]})

        if isinstance(result, dict):
            raw_messages = result.get("messages") or []
            struct_resp = result.get("structured_response")
            if struct_resp is not None and hasattr(struct_resp, "agent_answer"):
                final_answer = struct_resp.agent_answer or ""
        else:
            # 极少数情况下直接返回结构体
            if hasattr(result, "agent_answer"):
                final_answer = result.agent_answer or ""

        for m in raw_messages:
            cls_name = type(m).__name__
            msgs_dump.append(_msg_brief(m))
            if cls_name == "AIMessage":
                n_assistant_turns += 1
                tcs = getattr(m, "tool_calls", None) or []
                n_tool_calls += len(tcs)
                for tc in tcs:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name in by_name:
                        by_name[name] += 1

        if not final_answer:
            for m in reversed(raw_messages):
                if type(m).__name__ == "AIMessage" and getattr(m, "content", None):
                    final_answer = m.content
                    break

    except Exception as e:
        error = repr(e)
        traceback.print_exc()

    elapsed = time.time() - t0

    # API 调用估计：
    #   主 agent 每轮 thinking = 1 API call
    #   expand_and_keyword 内部 with_structured_output = 1 API call
    #   summary_related_doc 内部 with_structured_output = 1 API call
    #   retrieval_augment = 0 API call（Chroma 本地）
    api_calls_est = (
        n_assistant_turns
        + by_name["expand_and_keyword"]
        + by_name["summary_related_doc"]
    )

    return {
        "version": "v1",
        "query": query,
        "final_answer": final_answer,
        "latency_sec": round(elapsed, 3),
        "n_tool_calls": n_tool_calls,
        "n_assistant_turns": n_assistant_turns,
        "tool_calls_by_name": by_name,
        "api_calls_est": api_calls_est,
        "messages": msgs_dump,
        "error": error,
    }


# ===========================================================================
# v2 runner
# ===========================================================================


def run_v2(query: str, llm, max_turns: int = 5) -> dict:
    """跑 v2 Pure ReAct loop（仅 retrieval 一个工具）。"""
    t0 = time.time()
    error = None
    n_tool_calls = 0
    n_assistant_turns = 0
    final_answer = ""
    msgs_dump: list = []

    try:
        llm_with_tools = llm.bind_tools([retrieval_augment])
        msgs: list = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        for turn in range(max_turns):
            response: AIMessage = llm_with_tools.invoke(msgs)
            msgs.append(response)
            n_assistant_turns += 1

            if not getattr(response, "tool_calls", None):
                final_answer = response.content or ""
                break

            for tc in response.tool_calls:
                n_tool_calls += 1
                tool_name = tc["name"]
                args = tc.get("args", {}) or {}
                if tool_name == "retrieval_augment":
                    res = retrieval_augment.invoke(args)
                else:
                    res = f"Error: unknown tool {tool_name}"
                msgs.append(
                    ToolMessage(
                        content=str(res),
                        tool_call_id=tc.get("id") or f"call_{turn}_{n_tool_calls}",
                    )
                )
        else:
            # 循环跑满未给 final answer：取最后一条 AIMessage
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    final_answer = m.content
                    break

        msgs_dump = [_msg_brief(m) for m in msgs]

    except Exception as e:
        error = repr(e)
        traceback.print_exc()

    elapsed = time.time() - t0

    # v2 API 调用 = assistant 轮次（retrieval 本地 0 调用）
    api_calls_est = n_assistant_turns

    return {
        "version": "v2",
        "query": query,
        "final_answer": final_answer,
        "latency_sec": round(elapsed, 3),
        "n_tool_calls": n_tool_calls,
        "n_assistant_turns": n_assistant_turns,
        "tool_calls_by_name": {"retrieval_augment": n_tool_calls},
        "api_calls_est": api_calls_est,
        "messages": msgs_dump,
        "error": error,
    }


# ===========================================================================
# 输入加载
# ===========================================================================


def load_rows(queries_file: str, trajs_file: str) -> list[dict]:
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
        raise SystemExit("--queries 和 --trajs 都没找到。")
    return rows


# ===========================================================================
# 聚合 + 输出
# ===========================================================================


def _mean(records: list[dict], key: str) -> Optional[float]:
    vals = [
        r[key]
        for r in records
        if r.get(key) is not None and r.get("error") is None
    ]
    if not vals:
        return None
    return statistics.mean(vals)


def _success_rate(records: list[dict]) -> Optional[float]:
    ts = [
        r["task_success"]
        for r in records
        if r.get("task_success") is not None and r.get("error") is None
    ]
    if not ts:
        return None
    return sum(1 for x in ts if x) / len(ts)


def make_summary(v1: list[dict], v2: list[dict], teacher_model: str, n: int) -> str:
    metrics = [
        ("平均工具调用次数", "n_tool_calls", 2),
        ("平均 assistant 轮次", "n_assistant_turns", 2),
        ("平均外部 API 调用次数", "api_calls_est", 2),
        ("平均端到端 latency (s)", "latency_sec", 3),
    ]

    md = []
    md.append(f"# v1 vs v2 端到端对比")
    md.append("")
    md.append(f"- 主 LLM：`{teacher_model}`（两边相同，保证公平）")
    md.append(f"- 样本数：{n} 条 query")
    md.append(f"- v1 错误数：{sum(1 for r in v1 if r.get('error'))}")
    md.append(f"- v2 错误数：{sum(1 for r in v2 if r.get('error'))}")
    md.append("")
    md.append("## 核心指标对比")
    md.append("")
    md.append("| 指标 | v1（LangChain 3 工具） | v2（Pure ReAct 1 工具） | 改善 |")
    md.append("|------|------------------------|-------------------------|------|")

    for name, key, prec in metrics:
        v1v = _mean(v1, key)
        v2v = _mean(v2, key)
        if v1v is None or v2v is None:
            md.append(f"| {name} | - | - | N/A |")
            continue
        if v1v != 0:
            change = (v2v - v1v) / v1v * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "-"
        fmt = f"{{:.{prec}f}}"
        md.append(f"| {name} | {fmt.format(v1v)} | {fmt.format(v2v)} | {change_str} |")

    sr1 = _success_rate(v1)
    sr2 = _success_rate(v2)
    if sr1 is not None and sr2 is not None:
        delta = (sr2 - sr1) * 100
        md.append(f"| 任务成功率 | {sr1*100:.1f}% | {sr2*100:.1f}% | {delta:+.1f}pp |")
    else:
        md.append(f"| 任务成功率 | (无 ground_truth) | (无 ground_truth) | N/A |")

    md.append("")
    md.append("## 工具调用分布")
    md.append("")
    md.append("v1 工具调用次数（按工具名累计）：")
    by_name_total = {"expand_and_keyword": 0, "retrieval_augment": 0, "summary_related_doc": 0}
    for r in v1:
        for k, v in (r.get("tool_calls_by_name") or {}).items():
            by_name_total[k] = by_name_total.get(k, 0) + v
    md.append("")
    md.append("| 工具 | 总调用次数 | 平均/query |")
    md.append("|------|-----------|-----------|")
    n_v1 = max(1, len(v1))
    for k, v in by_name_total.items():
        md.append(f"| `{k}` | {v} | {v / n_v1:.2f} |")

    md.append("")
    md.append("v2 仅有 `retrieval_augment` 一个工具：")
    n_retrieval_v2 = sum(
        (r.get("tool_calls_by_name") or {}).get("retrieval_augment", 0) for r in v2
    )
    n_v2 = max(1, len(v2))
    md.append(f"- 总调用次数：{n_retrieval_v2}")
    md.append(f"- 平均/query：{n_retrieval_v2 / n_v2:.2f}")

    md.append("")
    md.append("## 简历量化用语建议")
    md.append("")
    md.append("基于上述结果，简历可以这样写：")
    md.append("")
    if v1v := _mean(v1, "n_tool_calls"):
        if v2v := _mean(v2, "n_tool_calls"):
            change = (v2v - v1v) / v1v * 100
            md.append(
                f"- 工具集精简后平均工具调用次数从 **{v1v:.1f} 次/query 降至 {v2v:.1f} 次/query**（{change:+.0f}%）"
            )
    if v1v := _mean(v1, "api_calls_est"):
        if v2v := _mean(v2, "api_calls_est"):
            change = (v2v - v1v) / v1v * 100
            md.append(
                f"- 外部 API 调用次数从 **{v1v:.1f} 次/query 降至 {v2v:.1f} 次/query**（{change:+.0f}%），"
                f"训练 rollout 完全脱离 DashScope API 依赖"
            )
    if v1v := _mean(v1, "latency_sec"):
        if v2v := _mean(v2, "latency_sec"):
            change = (v2v - v1v) / v1v * 100
            md.append(
                f"- 端到端响应延迟从 **{v1v:.2f}s 降至 {v2v:.2f}s**（{change:+.0f}%）"
            )

    return "\n".join(md) + "\n"


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="v1 vs v2 架构端到端对比")
    parser.add_argument("--queries", default="./data/eval/queries.txt")
    parser.add_argument(
        "--trajs",
        default="",
        help="可选：旧 trajs.jsonl 提供 ground_truth + key_evidence",
    )
    parser.add_argument("--output_dir", default="./data/eval/v1_vs_v2")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 query")
    parser.add_argument(
        "--teacher_model",
        default="qwen-plus",
        help="两边都用的主 LLM（保证对比公平）",
    )
    parser.add_argument("--max_turns_v2", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("请设置 QWEN_API_KEY")

    rows = load_rows(args.queries, args.trajs or "")
    if args.limit:
        rows = rows[: args.limit]
    print(f"对比 {len(rows)} 条 query，主 LLM={args.teacher_model}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_results: list[dict] = []
    v2_results: list[dict] = []

    for i, row in enumerate(rows, 1):
        q = row["query"]
        gt = row["ground_truth"]
        kws = row["keywords"]
        print(f"  [{i:03d}/{len(rows)}] {q[:60]}{'...' if len(q) > 60 else ''}")

        # v1
        try:
            llm_v1 = ChatTongyi(
                model=args.teacher_model, dashscope_api_key=api_key, temperature=0.0
            )
            v1 = run_v1(q, llm_v1)
        except Exception as e:
            v1 = {"version": "v1", "query": q, "error": repr(e)}
        v1["task_success"] = task_match(v1.get("final_answer", ""), gt, kws)
        v1_results.append(v1)

        # v2
        try:
            llm_v2 = ChatTongyi(
                model=args.teacher_model, dashscope_api_key=api_key, temperature=0.0
            )
            v2 = run_v2(q, llm_v2, max_turns=args.max_turns_v2)
        except Exception as e:
            v2 = {"version": "v2", "query": q, "error": repr(e)}
        v2["task_success"] = task_match(v2.get("final_answer", ""), gt, kws)
        v2_results.append(v2)

        v1_summary = (
            f"      v1: tools={v1.get('n_tool_calls', '?')}, "
            f"api={v1.get('api_calls_est', '?')}, "
            f"latency={v1.get('latency_sec', '?')}s, "
            f"success={v1.get('task_success')}"
        )
        v2_summary = (
            f"      v2: tools={v2.get('n_tool_calls', '?')}, "
            f"api={v2.get('api_calls_est', '?')}, "
            f"latency={v2.get('latency_sec', '?')}s, "
            f"success={v2.get('task_success')}"
        )
        print(v1_summary)
        print(v2_summary)

    # 写盘
    with open(out_dir / "v1_results.jsonl", "w", encoding="utf-8") as f:
        for r in v1_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "v2_results.jsonl", "w", encoding="utf-8") as f:
        for r in v2_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = make_summary(v1_results, v2_results, args.teacher_model, len(rows))
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print("=" * 60)
    print(summary)
    print(f"\n详细结果：{out_dir}/")


if __name__ == "__main__":
    main()
