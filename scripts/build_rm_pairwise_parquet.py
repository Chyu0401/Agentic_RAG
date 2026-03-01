"""
构建奖励模型（RM）训练用的 pairwise parquet：prompt, chosen, rejected。
可用于后续单独训练 RM（ranking loss），训练好的 RM 再在 GRPO 中通过 reward_model.enable=True 使用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RM 训练集：prompt, chosen, rejected")
    parser.add_argument(
        "--trajs",
        type=str,
        default="data/verl_trajs/trajs_verl_sft.jsonl",
        help="轨迹 JSONL（每行含 query、messages 或 final_answer），用于构造 chosen；rejected 需另行标注或用占位",
    )
    parser.add_argument(
        "--pairwise",
        type=str,
        default="",
        help="若已有 pairwise JSONL（每行 prompt, chosen, rejected），直接转 parquet",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/verl_trajs/rm_pairwise_train.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--rejected-placeholder",
        type=str,
        default="",
        help="无真实 rejected 时使用的占位回复（否则从 trajs 随机选一条当 rejected）",
    )
    args = parser.parse_args()

    if args.pairwise and Path(args.pairwise).exists():
        rows = []
        with open(args.pairwise, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (obj.get("prompt") or obj.get("query") or "").strip()
                chosen = (obj.get("chosen") or obj.get("response") or "").strip()
                rejected = (obj.get("rejected") or "").strip()
                if prompt and (chosen or rejected):
                    rows.append({"prompt": prompt, "chosen": chosen or "", "rejected": rejected or ""})
        if not rows:
            raise SystemExit("pairwise 文件未解析出有效行。")
        df = pd.DataFrame(rows)
    else:
        if not Path(args.trajs).exists():
            raise SystemExit("未提供 --pairwise 且 trajs 文件不存在，请指定数据源。")
        # 从轨迹构造：chosen = final_answer 或最后一条 assistant；rejected = 占位或随机另一条
        records = []
        with open(args.trajs, "r", encoding="utf-8") as f:
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
                chosen = obj.get("final_answer") or ""
                if not chosen and "messages" in obj:
                    for m in reversed(obj["messages"]):
                        if (m.get("role") or "").lower() == "assistant":
                            chosen = (m.get("content") or "").strip()
                            break
                chosen = (chosen or "").strip()
                rejected = args.rejected_placeholder or "抱歉，我暂时无法回答这个问题。"
                records.append({"prompt": query, "chosen": chosen, "rejected": rejected})
        if not records:
            raise SystemExit("未从 trajs 解析出有效样本。")
        df = pd.DataFrame(records)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"RM pairwise: {len(df)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
