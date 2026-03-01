"""
构建 GRPO 用的 prompt parquet。
每行：prompt（用户问题）、data_source（用于选择 reward）、reward_model（含 ground_truth 供 compute_score 使用）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_SOURCE = "agentic_rag"


def load_queries(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_trajs_verl_jsonl(path: str) -> list[dict]:
    """每条取 query 与 final_answer 作为 ground_truth（无标注时用模型答案当参考）。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = (obj.get("query") or "").strip()
            if not query:
                continue
            # 优先用 final_answer，否则从 messages 里取最后一条 assistant
            gt = obj.get("final_answer") or ""
            if not gt and "messages" in obj:
                for m in reversed(obj["messages"]):
                    if (m.get("role") or "").lower() == "assistant":
                        gt = (m.get("content") or "").strip()
                        break
            out.append({"query": query, "ground_truth": gt or ""})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 GRPO 使用的 prompt parquet")
    parser.add_argument(
        "--queries",
        type=str,
        default="data/eval/queries.txt",
        help="每行一个问题的文本文件",
    )
    parser.add_argument(
        "--trajs",
        type=str,
        default="",
        help="可选：trajs_verl_sft.jsonl，用于带 ground_truth 的 parquet（query + final_answer）",
    )
    parser.add_argument(
        "--out-train",
        type=str,
        default="data/verl_trajs/grpo_prompts_train.parquet",
        help="训练集 parquet 输出路径",
    )
    parser.add_argument(
        "--out-val",
        type=str,
        default="data/verl_trajs/grpo_prompts_val.parquet",
        help="验证集 parquet 输出路径",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trajs and Path(args.trajs).exists():
        rows = load_trajs_verl_jsonl(args.trajs)
        prompts = [r["query"] for r in rows]
        ground_truths = [r["ground_truth"] for r in rows]
    else:
        prompts = load_queries(args.queries)
        ground_truths = [""] * len(prompts)

    n = len(prompts)
    if n == 0:
        raise SystemExit("没有读到任何 prompt，请检查 --queries 或 --trajs。")

    rng = __import__("random").Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * args.train_fraction)
    train_idx = set(indices[:split])
    val_idx = set(indices[split:])

    def build_df(idx_set: set[int]) -> pd.DataFrame:
        data = []
        for i in idx_set:
            # VeRL RLHFDataset 期望 prompt 为 messages 列表，用于 apply_chat_template
            data.append({
                "prompt": [{"role": "user", "content": prompts[i]}],
                "data_source": DATA_SOURCE,
                "reward_model": {"ground_truth": ground_truths[i]},
            })
        return pd.DataFrame(data)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)

    train_df = build_df(train_idx)
    val_df = build_df(val_idx)
    train_df.to_parquet(args.out_train, index=False)
    val_df.to_parquet(args.out_val, index=False)
    print(f"Train: {len(train_df)} -> {args.out_train}")
    print(f"Val:   {len(val_df)} -> {args.out_val}")


if __name__ == "__main__":
    main()
