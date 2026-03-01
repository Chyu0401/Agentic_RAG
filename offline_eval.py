import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from src.llm import get_qwen_chat
from src.tools.query_preprocess import expand_and_keyword
from src.tools.reading import SummaryRelatedDocFormat, summary_related_doc
from src.tools.retrieval import retrieval_augment


def load_queries(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"未找到问题文件：{path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [q for q in lines if q]


class IrrelevantRatioFormat(BaseModel):
    """
    LLM 评估结果：当前上下文中与 query 无关内容的大致比例（0~1）。
    """

    irrelevant_ratio: float = Field(
        ...,
        description="上下文中与 query 无关内容所占比例（0~1 之间的小数）。",
    )


def estimate_irrelevant_ratio(
    llm: BaseChatModel,
    query: str,
    context: str,
) -> float:
    """
    使用 LLM 评估“无关内容比例”。为了节省成本，可以在命令行参数里关掉。
    """
    if not context.strip():
        return 0.0

    system_prompt = (
        "你是一个严谨的评估助手。现在给你一个用户问题（query）和一段上下文（context）。\n"
        "请你判断：在这段 context 中，**大约**有多少比例的内容与 query 无关？\n"
        "只需输出一个 0~1 之间的小数，例如 0.3 表示大约 30% 内容无关。\n"
    )
    messages = [
        ("system", system_prompt),
        (
            "human",
            # 为防止过长导致结构化解析异常，只截取前 2000 字符
            f"用户的 query：{query}\n\n上下文（context）：\n{context[:2000]}",
        ),
    ]
    try:
        structured_llm = llm.with_structured_output(IrrelevantRatioFormat)
        res = structured_llm.invoke(messages)
    except Exception as e:
        return 0.0

    if res is None:
        return 0.0

    # 做一下截断保护
    return float(max(0.0, min(1.0, res.irrelevant_ratio)))


@dataclass
class EvalRecord:
    query: str
    expand_query: str
    keyword: str
    baseline_context: str
    agentic_context: str
    baseline_length: int
    agentic_length: int
    baseline_irrelevant_ratio: Optional[float] = None
    agentic_irrelevant_ratio: Optional[float] = None


def run_eval(
    questions_path: str,
    output_path: str,
    with_irrelevant_score: bool = True,
) -> None:
    llm = get_qwen_chat()
    queries = load_queries(questions_path)
    if not queries:
        raise RuntimeError(f"从 {questions_path} 中未读取到任何问题。")

    records: List[EvalRecord] = []

    for q in queries:
        # 1) Query 预处理：扩写 + 关键词
        ek_res = expand_and_keyword(llm=llm, query=q)
        expand_q = ek_res.expand_query
        keyword = ek_res.keyword

        # 2) Baseline-RAG：只用 retrieval_augment 的结果作为上下文
        baseline_context: str = retrieval_augment.invoke(
            {"query": expand_q, "keyword": keyword}
        )

        # 3) Agentic-RAG：在 baseline_context 上再做精读筛选
        summary_res: SummaryRelatedDocFormat = summary_related_doc(
            llm=llm,
            query=expand_q,
            related_doc=baseline_context,
        )
        agentic_context = summary_res.summary_related_doc_res

        baseline_len = len(baseline_context)
        agentic_len = len(agentic_context)

        if with_irrelevant_score:
            base_irrel = estimate_irrelevant_ratio(llm, q, baseline_context)
            agent_irrel = estimate_irrelevant_ratio(llm, q, agentic_context)
        else:
            base_irrel = None
            agent_irrel = None

        records.append(
            EvalRecord(
                query=q,
                expand_query=expand_q,
                keyword=keyword,
                baseline_context=baseline_context,
                agentic_context=agentic_context,
                baseline_length=baseline_len,
                agentic_length=agentic_len,
                baseline_irrelevant_ratio=base_irrel,
                agentic_irrelevant_ratio=agent_irrel,
            )
        )

    # 统计指标
    n = len(records)
    avg_base_len = sum(r.baseline_length for r in records) / n
    avg_agent_len = sum(r.agentic_length for r in records) / n
    length_reduction = 1.0 - (avg_agent_len / avg_base_len) if avg_base_len > 0 else 0.0

    if with_irrelevant_score:
        avg_base_irrel = (
            sum(r.baseline_irrelevant_ratio or 0.0 for r in records) / n
        )
        avg_agent_irrel = (
            sum(r.agentic_irrelevant_ratio or 0.0 for r in records) / n
        )
        irrel_reduction = (
            1.0 - (avg_agent_irrel / avg_base_irrel) if avg_base_irrel > 0 else 0.0
        )
    else:
        avg_base_irrel = None
        avg_agent_irrel = None
        irrel_reduction = None

    summary = {
        "num_queries": n,
        "avg_baseline_length": avg_base_len,
        "avg_agentic_length": avg_agent_len,
        "length_reduction_ratio": length_reduction,
        "avg_baseline_irrelevant_ratio": avg_base_irrel,
        "avg_agentic_irrelevant_ratio": avg_agent_irrel,
        "irrelevant_reduction_ratio": irrel_reduction,
    }

    out = {
        "summary": summary,
        "records": [asdict(r) for r in records],
    }

    Path(output_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 评估完成 ===")
    print(f"问题数量: {n}")
    print(
        f"平均上下文长度: baseline={avg_base_len:.1f}, "
        f"agentic={avg_agent_len:.1f}, "
        f"降低比例={length_reduction*100:.1f}%"
    )
    if with_irrelevant_score and avg_base_irrel is not None:
        print(
            f"平均无关内容比例: baseline={avg_base_irrel:.3f}, "
            f"agentic={avg_agent_irrel:.3f}, "
            f"降低比例={irrel_reduction*100:.1f}%"
        )
    print(f"详细结果已写入: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比有/无精读 Agent 时的上下文质量（长度 & 无关内容比例）"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="./data/eval/queries.txt",
        help="校规相关问题文件路径，每行一个问题。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/eval/offline_eval_result.json",
        help="评估结果输出路径（JSON）。",
    )
    parser.add_argument(
        "--no_irrelevant_score",
        action="store_true",
        help="不计算无关内容比例（只统计长度），可节省 LLM 调用成本。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        questions_path=args.questions_file,
        output_path=args.output_file,
        with_irrelevant_score=not args.no_irrelevant_score,
    )

