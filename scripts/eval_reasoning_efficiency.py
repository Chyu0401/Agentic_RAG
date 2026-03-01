#!/usr/bin/env python3
"""
计算「复杂 Query 推理效率提升」百分比，用于复现简历中的 21.2% 等数值。

定义：
  - 推理效率 = 1 / 平均工具调用步数（仅答对样本）
  - 推理效率提升 = (avg_steps_before - avg_steps_after) / avg_steps_after
  - 仅统计「复杂 Query」：由 --complex_queries 指定，或全部 query

用法：
  # 用两份轨迹 JSONL（基线与 GRPO 后），每行含 query、response、ground_truth
  python scripts/eval_reasoning_efficiency.py \
    --baseline data/verl_trajs/baseline_trajs.jsonl \
    --after data/verl_trajs/grpo_trajs.jsonl

  # 仅统计复杂 Query（每行一个 query）
  python scripts/eval_reasoning_efficiency.py \
    --baseline data/verl_trajs/baseline_trajs.jsonl \
    --after data/verl_trajs/grpo_trajs.jsonl \
    --complex_queries data/eval/queries_complex.txt

JSONL 每行格式示例（至少包含）：
  {"query": "用户问题", "response": "完整模型输出（含 Action/Observation 等）", "ground_truth": "参考答案"}
  或 "reward_model": {"ground_truth": "参考答案"} 代替 ground_truth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 从同目录的 reward_rag 复用步数统计与任务成功判定（需先加 scripts 到 path）
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from reward_rag import (
    REWARD_TASK_SUCCESS,
    _count_steps_and_invalid_calls,
    compute_score,
)


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def get_ground_truth(row: dict) -> str | dict:
    if "ground_truth" in row:
        return row["ground_truth"]
    rm = row.get("reward_model") or {}
    return rm.get("ground_truth", "")


def get_response(row: dict) -> str:
    if "response" in row and row["response"]:
        return row["response"]
    if "solution_str" in row and row["solution_str"]:
        return row["solution_str"]
    # 由 trajectory + final_answer 拼成近似完整输出（便于步数统计）
    history = row.get("trajectory") or row.get("history") or []
    final = row.get("final_answer") or ""
    parts = []
    for step in history:
        action = step.get("action") or step.get("tool_call") or {}
        obs = step.get("observation") or step.get("result") or ""
        name = action.get("name") or action.get("tool") or "tool"
        args = action.get("args") or action.get("arguments") or ""
        parts.append(f"Action: {name} {args}".strip())
        parts.append(f"Observation: {obs}")
    if final:
        parts.append(f"Answer: {final}")
    return "\n".join(parts) if parts else final


def load_complex_queries(path: str | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算复杂 Query 上推理效率提升（用于复现简历 21.2% 等）"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="基线模型轨迹 JSONL（SFT-only 或 GRPO 前）",
    )
    parser.add_argument(
        "--after",
        type=str,
        required=True,
        help="GRPO 后模型轨迹 JSONL",
    )
    parser.add_argument(
        "--complex_queries",
        type=str,
        default=None,
        help="复杂 Query 列表，每行一问；不传则使用全部 query",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        default=True,
        help="仅统计在基线和 after 中都答对的 query（默认 True）",
    )
    parser.add_argument(
        "--no_paired",
        action="store_true",
        help="不要求成对答对，分别算两边的平均步数（可能集合不同）",
    )
    args = parser.parse_args()
    if args.no_paired:
        args.paired = False

    complex_set = load_complex_queries(args.complex_queries)
    baseline_rows = load_jsonl(args.baseline)
    after_rows = load_jsonl(args.after)

    def row_key(r: dict) -> str:
        return (r.get("query") or "").strip()

    baseline_by_query = {row_key(r): r for r in baseline_rows}
    after_by_query = {row_key(r): r for r in after_rows}
    common_queries = set(baseline_by_query) & set(after_by_query)
    if complex_set:
        common_queries &= complex_set
        print(f"复杂 Query 数量（与两份轨迹交集）: {len(common_queries)}")
    else:
        print(f"共同 Query 数量: {len(common_queries)}")

    def is_correct(row: dict) -> bool:
        gt = get_ground_truth(row)
        sol = get_response(row)
        r = compute_score(
            data_source="agentic_rag",
            solution_str=sol,
            ground_truth=gt,
        )
        return r >= REWARD_TASK_SUCCESS

    def steps(row: dict) -> int:
        return _count_steps_and_invalid_calls(get_response(row))[0]

    # 收集答对且步数
    baseline_correct = {}  # query -> steps
    after_correct = {}
    for q in common_queries:
        b_row = baseline_by_query.get(q)
        a_row = after_by_query.get(q)
        if b_row and is_correct(b_row):
            baseline_correct[q] = steps(b_row)
        if a_row and is_correct(a_row):
            after_correct[q] = steps(a_row)

    if args.paired:
        paired_queries = set(baseline_correct) & set(after_correct)
        if not paired_queries:
            print("没有在基线和 GRPO 后都答对的 query，无法计算成对效率提升。")
            return
        steps_before = [baseline_correct[q] for q in paired_queries]
        steps_after = [after_correct[q] for q in paired_queries]
        n = len(paired_queries)
        avg_before = sum(steps_before) / n
        avg_after = sum(steps_after) / n
        print(f"成对答对 query 数: {n}")
    else:
        if not baseline_correct or not after_correct:
            print("基线或 GRPO 后答对样本为空，无法计算。")
            return
        avg_before = sum(baseline_correct.values()) / len(baseline_correct)
        avg_after = sum(after_correct.values()) / len(after_correct)
        print(f"基线答对数: {len(baseline_correct)}, GRPO 后答对数: {len(after_correct)}")

    # 推理效率提升 = (avg_before - avg_after) / avg_after
    if avg_after <= 0:
        print("avg_steps_after 为 0，无法计算提升。")
        return
    improvement = (avg_before - avg_after) / avg_after
    pct = improvement * 100

    print(f"基线平均步数（答对样本）: {avg_before:.2f}")
    print(f"GRPO 后平均步数（答对样本）: {avg_after:.2f}")
    print(f"推理效率提升 (avg_before - avg_after) / avg_after = {pct:.2f}%")
    print(f"（简历可写：复杂 Query 推理效率提升 {pct:.1f}%）")


if __name__ == "__main__":
    main()
