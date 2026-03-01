#!/usr/bin/env python3
"""
将 SFT 用 JSONL 按比例划分为训练集与验证集（默认 90% / 10%），便于在 yaml 中配置 train_files / val_files。

用法：
  python split_sft_train_val.py
  python split_sft_train_val.py --input ./data/verl_trajs/trajs_verl_sft.jsonl --val_ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="划分 SFT JSONL 为 train/val")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/verl_trajs/trajs_verl_sft.jsonl",
        help="完整 SFT JSONL 路径",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例，默认 0.1（10%%）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机划分种子",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认与 input 同目录",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"输入文件不存在: {inp}")

    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "trajs_verl_sft_train.jsonl"
    val_path = out_dir / "trajs_verl_sft_val.jsonl"

    lines = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    random.seed(args.seed)
    random.shuffle(lines)
    n = len(lines)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    with train_path.open("w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"总数 {n} -> 训练 {n_train} ({train_path}), 验证 {n_val} ({val_path})")


if __name__ == "__main__":
    main()
