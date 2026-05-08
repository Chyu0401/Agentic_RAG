"""把 rejection-sampling 出来的 ReAct 轨迹（multi-turn）转成 VeRL multi-turn SFT 用的 parquet。

输入：scripts/build_sft_trajectories_react.py 的输出（JSONL，含 messages/tool_calls/tool 完整多轮 chat）
输出：parquet，含 `messages` 列（OpenAI chat 格式）+ `tools` 列（function schema 列表）

关键说明：
1. VeRL 的 MultiTurnSFTDataset 调 tokenizer.apply_chat_template(messages, tools=tools)
   → Qwen3 的 chat template 会把 tools schema 注入到 system prompt 里（<tools>JSON</tools> 段）
   → 训练时模型学到的格式 = inference (GRPO rollout) 时模型看到的格式（rollout 里 tools 也是同一份）
2. 我们把 retrieval_augment 的 schema 直接 hardcode 在这个脚本里，与 scripts/tool_config/rag_tools.yaml 保持一致
   （改动 schema 时两边都要改；后续可以考虑统一从 yaml 读）
3. loss mask 由 MultiTurnSFTDataset 自动生成（assistant token = 1，其余 = 0），不需要我们管
4. 训练-验证集划分在这里做（默认 9:1）

用法：
    python scripts/convert_react_trajs_to_verl_format.py \
        --input data/verl_trajs/trajs_react_filtered.jsonl \
        --out_train data/verl_trajs/sft_react_train.parquet \
        --out_val data/verl_trajs/sft_react_val.parquet
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# Retrieval tool 的 OpenAI function schema —— 与 tool_config/rag_tools.yaml 保持一致
# Qwen3 chat template 接收的 tools 字段就是这种 list[dict] 格式
RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieval_augment",
        "description": (
            "Search the school regulation knowledge base. Returns the top-5 most relevant "
            "document chunks. Use this tool when the user's question is about school rules, "
            "policies, or any specific regulation that requires factual grounding from the "
            "knowledge base. You may call this tool multiple times with different (refined) "
            "queries if the first results are not sufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query to retrieve relevant documents. You may rewrite the "
                        "user's original question into a more retrieval-friendly form."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "A short, specific keyword extracted from the query, used for "
                        "keyword-aware reranking. Pass empty string if no clear keyword."
                    ),
                },
            },
            "required": ["query", "keyword"],
        },
    },
}

TOOLS_LIST = [RETRIEVAL_TOOL_SCHEMA]


def _normalize_tool_calls(tool_calls):
    """把 tool_calls 字段标准化成 chat template 接受的 list[dict] 格式。

    chat template 要求每条 tool_call 是：
        {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
    其中 arguments 是 JSON 字符串（不是 dict）。
    """
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        arguments = fn.get("arguments") or tc.get("arguments") or "{}"
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments = "{}"
        out.append(
            {
                "id": tc.get("id") or "",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return out or None


def _normalize_message(m: dict) -> dict | None:
    """规范一条 message 为 chat template 接受的格式。"""
    if not isinstance(m, dict):
        return None
    role = m.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        return None

    content = m.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        # 多模态 content（list of dict）暂不支持，强制转字符串避免 schema 错乱
        try:
            content = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(content)

    out: dict = {"role": role, "content": content}

    if role == "assistant":
        tcs = _normalize_tool_calls(m.get("tool_calls"))
        if tcs:
            out["tool_calls"] = tcs

    if role == "tool":
        # Qwen3 chat template 期望 tool 消息有 tool_call_id（关联到上一条 assistant 的 tool_call）
        tcid = m.get("tool_call_id") or ""
        out["tool_call_id"] = tcid

    return out


def _validate_messages(messages: list[dict]) -> bool:
    """对一条多轮 chat 做基本结构验证，返回是否有效。

    要求：
    - 至少 3 条：system + user + assistant
    - 必须有至少一条 assistant 消息
    - 若有 tool 消息，前一条必须是带 tool_calls 的 assistant
    """
    if not messages or len(messages) < 3:
        return False
    if not any(m["role"] == "assistant" for m in messages):
        return False
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            if i == 0:
                return False
            prev = messages[i - 1]
            if prev["role"] != "assistant" or not prev.get("tool_calls"):
                # 中间隔了多条 tool 也算正常（一个 assistant 可能 emit 多个 tool_call）
                # 往前回溯找最近的 assistant
                j = i - 1
                while j >= 0 and messages[j]["role"] == "tool":
                    j -= 1
                if j < 0 or messages[j]["role"] != "assistant" or not messages[j].get("tool_calls"):
                    return False
    return True


def convert_one(raw: dict) -> dict | None:
    """单条 JSONL → parquet 行。"""
    raw_messages = raw.get("messages") or []
    normed = []
    for m in raw_messages:
        nm = _normalize_message(m)
        if nm is not None:
            normed.append(nm)

    if not _validate_messages(normed):
        return None

    return {
        "messages": normed,
        "tools": TOOLS_LIST,
        # 保留几个 metadata 列便于事后分析；MultiTurnSFTDataset 会忽略未配置的列
        "query": raw.get("query", ""),
        "reward": raw.get("reward"),
        "n_tool_calls": raw.get("n_tool_calls"),
        "task_success": raw.get("task_success"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReAct 多轮轨迹 JSONL → VeRL multi-turn SFT parquet"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/verl_trajs/trajs_react_filtered.jsonl",
        help="build_sft_trajectories_react.py 的输出 JSONL",
    )
    parser.add_argument(
        "--out_train",
        type=str,
        default="./data/verl_trajs/sft_react_train.parquet",
    )
    parser.add_argument(
        "--out_val",
        type=str,
        default="./data/verl_trajs/sft_react_val.parquet",
    )
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dedup_by_query",
        action="store_true",
        help="同一 query 多个采样保留 reward 最高的一条（默认全保留以扩大数据量）",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"输入文件不存在：{inp}")

    rows: list[dict] = []
    n_raw, n_invalid = 0, 0
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                n_invalid += 1
                continue
            converted = convert_one(raw)
            if converted is None:
                n_invalid += 1
                continue
            rows.append(converted)

    if not rows:
        raise SystemExit("没有有效轨迹可写。检查 --input。")

    if args.dedup_by_query:
        # 同一 query 留 reward 最高的一条
        best: dict[str, dict] = {}
        for r in rows:
            q = r.get("query", "")
            cur = best.get(q)
            if cur is None or (r.get("reward") or 0) > (cur.get("reward") or 0):
                best[q] = r
        rows = list(best.values())

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    split = int(n * args.train_fraction)
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[-max(1, n // 20) :]  # 至少留点 val

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_parquet(args.out_train, index=False)
    pd.DataFrame(val_rows).to_parquet(args.out_val, index=False)

    print(f"输入 JSONL: {n_raw} 条，无效/丢弃: {n_invalid}")
    print(f"有效轨迹: {n}（dedup_by_query={args.dedup_by_query}）")
    print(f"Train: {len(train_rows)} -> {args.out_train}")
    print(f"Val:   {len(val_rows)} -> {args.out_val}")


if __name__ == "__main__":
    main()
