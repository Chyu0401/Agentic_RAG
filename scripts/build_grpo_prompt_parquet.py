"""构建 GRPO（multi-turn）用的 prompt parquet。

每行字段（与 VeRL multi-turn + return_raw_chat=True 兼容）：
- prompt: list[dict]，含 system + user 两条 message
- data_source: 用于 compute_score 路由
- reward_model: dict，含 ground_truth 供 compute_score 使用
- extra_info: dict，可选携带 key_evidence、原始 query 等

system prompt 采用 Pure ReAct 模式：
- 仅暴露一个工具 retrieval_augment(query, keyword)
- query 改写、关键词抽取、文档精读统统让模型自己在 thinking 里完成
- 不再把 expand_and_keyword / summary_related_doc 作为独立工具
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_SOURCE = "agentic_rag"

SYSTEM_PROMPT = """You are a helpful assistant for answering questions about university regulations.

You have access to ONE tool:
- `retrieval_augment(query: str, keyword: str)`: search the regulation knowledge base, returns top-5 relevant document chunks.

Follow this Pure ReAct procedure:
1. **Think first.** Before any tool call, briefly reason about whether the question needs the knowledge base. Off-topic or general-knowledge questions should be answered directly without retrieval.
2. **Plan the search.** If retrieval is needed, decide what query and keyword to use. You may rewrite the user's original question into a more retrieval-friendly form (expand abbreviations, add context terms, normalize phrasing). Extract one short, specific keyword for keyword-aware reranking.
3. **Call the tool.** Issue exactly one `retrieval_augment` call with your chosen query and keyword.
4. **Read and reason.** After the tool returns documents, identify which parts are actually relevant to the user's question. Quote or paraphrase the supporting evidence.
5. **Refine if needed.** If the first retrieval did not surface the answer, you may issue at most TWO more retrieval calls with refined queries (max 3 calls total). Do not call the tool with the same query repeatedly.
6. **Final answer.** Once you have enough evidence, output the final answer. If the knowledge base does not contain the answer, say so explicitly: "没有在文档中检索到相关内容，所以无法准确回答。"

Constraints:
- Do not invent or hallucinate document content. Only cite what `retrieval_augment` actually returned.
- Be concise. Long answers without grounding will be penalized.
- Maximum 3 tool calls per question. Extra calls incur efficiency penalty."""


def load_queries(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_trajs_verl_jsonl(path: str) -> list[dict]:
    """从 trajs_verl_sft.jsonl 取 query + final_answer（无标注时用模型答案当参考）。"""
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
            gt = obj.get("final_answer") or ""
            if not gt and "messages" in obj:
                for m in reversed(obj["messages"]):
                    if (m.get("role") or "").lower() == "assistant":
                        gt = (m.get("content") or "").strip()
                        break
            # 可选携带 key_evidence（如果 jsonl 里有的话）
            key_evidence = obj.get("key_evidence") or []
            keywords = obj.get("keywords") or []
            out.append(
                {
                    "query": query,
                    "ground_truth": gt or "",
                    "key_evidence": key_evidence,
                    "keywords": keywords,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 multi-turn GRPO 使用的 prompt parquet")
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
        help="可选：trajs_verl_sft.jsonl，用于带 ground_truth + key_evidence 的 parquet",
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
        if not rows:
            raise SystemExit(f"从 {args.trajs} 没读到任何有效 query。")
    else:
        prompts = load_queries(args.queries)
        rows = [
            {"query": q, "ground_truth": "", "key_evidence": [], "keywords": []}
            for q in prompts
        ]

    n = len(rows)
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
            row = rows[i]
            # multi-turn 需要 system + user 两条 message
            # ground_truth 走 dict 形式（reward 函数支持 answer + keywords）
            # 不在 ground_truth 里塞 key_evidence，统一通过 extra_info 传更干净
            gt_dict = {
                "answer": row["ground_truth"],
                "keywords": row.get("keywords", []),
            }
            extra_info = {
                "key_evidence": row.get("key_evidence", []),
                "raw_query": row["query"],
            }
            data.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["query"]},
                    ],
                    "data_source": DATA_SOURCE,
                    "reward_model": {"ground_truth": gt_dict},
                    "extra_info": extra_info,
                }
            )
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
