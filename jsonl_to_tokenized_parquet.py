#!/usr/bin/env python3
"""

- 输入：trajs_verl_sft.jsonl（每行含 messages 或 prompt/response）
- 输出：trajs_verl_sft_train.parquet、trajs_verl_sft_val.parquet（列：input_ids, attention_mask）

与参考实现一致：prompt + response 拼成整段再 tokenize，truncation + padding 到固定 max_length，
训练时由 dataloader 直接读 token 序列（不再在线调 tokenizer）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _get_prompt_response(obj: dict) -> tuple[str, str]:
    """从一条记录中取出 prompt 与 response。支持 messages 或 直接 prompt/response。"""
    if "messages" in obj:
        msgs = obj["messages"]
        prompt, response = "", ""
        for m in msgs:
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if role == "user":
                prompt = content
            elif role == "assistant":
                response = content
                break
        return prompt, response
    return (obj.get("prompt") or "").strip(), (obj.get("response") or "").strip()


def jsonl_to_tokenized_parquet(
    jsonl_path: str,
    parquet_train_path: str,
    parquet_val_path: str,
    tokenizer_name: str,
    max_length: int = 4096,
    train_fraction: float = 0.9,
    batch_size: int = 5000,
    seed: int = 42,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    random.seed(seed)
    train_buffer: list[dict] = []
    val_buffer: list[dict] = []
    train_count = 0
    val_count = 0
    schema = None

    # 先读一小批推断 schema
    sample_data: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, resp = _get_prompt_response(obj)
            full = prompt + resp
            enc = tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            )
            sample_data.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            })
            if len(sample_data) >= batch_size:
                break

    if not sample_data:
        raise ValueError(f"未从 {jsonl_path} 解析出任何有效样本，请检查格式。")

    table0 = pa.Table.from_pylist(sample_data)
    schema = table0.schema

    Path(parquet_train_path).parent.mkdir(parents=True, exist_ok=True)
    writer_train = pq.ParquetWriter(parquet_train_path, schema, compression="snappy")
    writer_val = pq.ParquetWriter(parquet_val_path, schema, compression="snappy")

    # 把 sample 按比例归入 train/val
    for record in sample_data:
        if random.random() < train_fraction:
            train_buffer.append(record)
            train_count += 1
        else:
            val_buffer.append(record)
            val_count += 1

    def flush_train():
        nonlocal train_buffer
        if len(train_buffer) >= batch_size:
            table = pa.Table.from_pylist(train_buffer, schema=schema)
            writer_train.write_table(table)
            train_buffer = []

    def flush_val():
        nonlocal val_buffer
        if len(val_buffer) >= batch_size:
            table = pa.Table.from_pylist(val_buffer, schema=schema)
            writer_val.write_table(table)
            val_buffer = []

    # 从第 batch_size 条之后继续读并处理
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < len(sample_data):
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, resp = _get_prompt_response(obj)
            full = prompt + resp
            enc = tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            )
            record = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
            if random.random() < train_fraction:
                train_buffer.append(record)
                train_count += 1
                flush_train()
            else:
                val_buffer.append(record)
                val_count += 1
                flush_val()

    if train_buffer:
        table = pa.Table.from_pylist(train_buffer, schema=schema)
        writer_train.write_table(table)
    if val_buffer:
        table = pa.Table.from_pylist(val_buffer, schema=schema)
        writer_val.write_table(table)

    writer_train.close()
    writer_val.close()

    print(f"Train 样本数：{train_count} -> {parquet_train_path}")
    print(f"Val 样本数：{val_count} -> {parquet_val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 SFT JSONL 分词并 90/10 划分，写出 train/val Parquet（与参考流程一致）"
    )
    parser.add_argument("--jsonl", type=str, default="./data/verl_trajs/trajs_verl_sft.jsonl")
    parser.add_argument("--parquet_train", type=str, default="./data/verl_trajs/trajs_verl_sft_train.parquet")
    parser.add_argument("--parquet_val", type=str, default="./data/verl_trajs/trajs_verl_sft_val.parquet")
    parser.add_argument("--tokenizer_name", type=str, required=True, default="/data/home/xmju/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554", help="与 SFT 基座一致的 HuggingFace tokenizer 名称，如 Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jsonl_to_tokenized_parquet(
        args.jsonl,
        args.parquet_train,
        args.parquet_val,
        args.tokenizer_name,
        args.max_length,
        args.train_fraction,
        args.batch_size,
        args.seed,
    )


if __name__ == "__main__":
    main()
