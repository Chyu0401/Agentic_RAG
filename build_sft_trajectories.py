#!/usr/bin/env python3
"""
构建 SFT 轨迹数据：从问题列表逐条跑 RAG Agent，将完整轨迹写入 JSONL。

用法：
  python build_sft_trajectories.py
  python build_sft_trajectories.py --queries_file ./data/eval/queries.txt --output_file ./data/verl_trajs/trajs.jsonl
  python build_sft_trajectories.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

from src.llm import get_qwen_chat
from src.agents.rag_agent_logging import RagAgentWithLogging


def load_queries(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"问题文件不存在: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [q for q in lines if q]


def load_finished_queries(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    finished: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # 为避免引入额外依赖，这里用最简单的 eval 替代 json 解析，
                # 但更安全的做法是直接 import json 再 json.loads。
                import json

                obj = json.loads(line)
                q = obj.get("query")
                if isinstance(q, str):
                    finished.add(q)
            except Exception:
                continue
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 Agent 工具调用轨迹，用于 VeRL SFT 数据。"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default="./data/eval/queries.txt",
        help="问题列表路径，每行一个问题。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/verl_trajs/trajs.jsonl",
        help="轨迹 JSONL 输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少条问题（用于试跑或分批）。不传则处理全部。",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="单条失败时跳过并继续，不中断整体。",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    if args.limit is not None:
        queries = queries[: args.limit]
    n = len(queries)
    if n == 0:
        print("未读取到任何问题，请检查 --queries_file。")
        sys.exit(1)

    finished_queries = load_finished_queries(args.output_file)
    if finished_queries:
        print(f"检测到已有 {len(finished_queries)} 条轨迹，未完成的将继续生成。")

    print(f"共 {n} 条问题，输出: {args.output_file}")
    llm = get_qwen_chat()
    agent = RagAgentWithLogging(llm=llm, log_path=args.output_file)

    ok, skip, fail = 0, 0, 0
    for i, q in enumerate(queries, start=1):
        if q in finished_queries:
            skip += 1
            print(f"  [{i}/{n}] SKIP(existing): {q[:50]}{'...' if len(q) > 50 else ''}")
            continue
        try:
            agent.run_and_log(q, task_success=True)
            ok += 1
            print(f"  [{i}/{n}] OK: {q[:50]}{'...' if len(q) > 50 else ''}")
        except Exception as e:
            fail += 1
            if args.skip_errors:
                print(f"  [{i}/{n}] SKIP: {q[:50]}... -> {e!r}")
            else:
                print(f"  [{i}/{n}] FAIL: {q[:50]}... -> {e!r}")
                raise

    print(f"\n完成: 成功 {ok}, 跳过(已存在) {skip}, 失败 {fail}, 轨迹已写入 {args.output_file}")


if __name__ == "__main__":
    main()
