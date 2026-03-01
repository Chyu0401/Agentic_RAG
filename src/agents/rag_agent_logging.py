from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from ..tools.query_preprocess import ExpandAndKeywordFormat, expand_and_keyword
from ..tools.reading import SummaryRelatedDocFormat, summary_related_doc
from ..tools.retrieval import retrieval_augment


@dataclass
class TrajectoryStep:
    """
    单步 Agent 轨迹，用于后续 VeRL SFT / RL 训练。
    """

    step: int
    observation: str
    action: Dict[str, Any]  # 包含 tool 名称、输入、输出等


@dataclass
class TrajectoryRecord:
    """
    一条完整的 Agent 轨迹。
    """

    query: str
    history: List[TrajectoryStep]
    final_answer: str
    task_success: Optional[bool] = None
    meta: Optional[Dict[str, Any]] = None


class RagAgentWithLogging:
    """
    基于当前 RAG 流程的显式控制器 + 轨迹记录器。

    流程固定为：
      1) expand_and_keyword（可视为 Query 预处理工具）
      2) retrieval_augment  （向量检索 + 关键词感知重排）
      3) summary_related_doc（精读筛选证据）
      4) LLM 基于证据生成最终回答

    每一步都会记录 observation / action（tool 名、输入、输出），
    并将整条轨迹以 JSONL 形式追加写入 log_path，便于后续构造 VeRL SFT 数据。
    """

    def __init__(self, llm: BaseChatModel, log_path: str = "./data/verl_trajs/trajs.jsonl") -> None:
        self.llm = llm
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_record(self, record: TrajectoryRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            json.dump(
                {
                    "query": record.query,
                    "history": [asdict(step) for step in record.history],
                    "final_answer": record.final_answer,
                    "task_success": record.task_success,
                    "meta": record.meta or {},
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    def run_and_log(
        self,
        query: str,
        task_success: Optional[bool] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        执行一轮固定的 RAG 流程，并记录完整轨迹到 JSONL 文件。

        返回：
            final_answer: 最终回答文本（同时也写入轨迹）。
        """
        history: List[TrajectoryStep] = []
        step_idx = 1

        # Step 1: Query 预处理（expand_and_keyword）
        ek_res: ExpandAndKeywordFormat = expand_and_keyword(llm=self.llm, query=query)
        history.append(
            TrajectoryStep(
                step=step_idx,
                observation=query,
                action={
                    "tool": "expand_and_keyword",
                    "tool_input": {"query": query},
                    "tool_output": {
                        "expand_query": ek_res.expand_query,
                        "keyword": ek_res.keyword,
                    },
                },
            )
        )
        step_idx += 1

        expand_q = ek_res.expand_query
        keyword = ek_res.keyword

        # Step 2: 检索增强（retrieval_augment）
        related_doc: str = retrieval_augment.invoke(
            {"query": expand_q, "keyword": keyword}
        )
        history.append(
            TrajectoryStep(
                step=step_idx,
                observation=f"expand_query={expand_q}, keyword={keyword}",
                action={
                    "tool": "retrieval_augment",
                    "tool_input": {"query": expand_q, "keyword": keyword},
                    "tool_output": related_doc,
                },
            )
        )
        step_idx += 1

        # Step 3: 精读筛选（summary_related_doc）
        summary_res: SummaryRelatedDocFormat = summary_related_doc(
            llm=self.llm,
            query=expand_q,
            related_doc=related_doc,
        )
        agentic_context = summary_res.summary_related_doc_res
        history.append(
            TrajectoryStep(
                step=step_idx,
                observation=related_doc,
                action={
                    "tool": "summary_related_doc",
                    "tool_input": {"query": expand_q, "related_doc": related_doc},
                    "tool_output": agentic_context,
                },
            )
        )
        step_idx += 1

        # Step 4: 基于证据生成最终回答
        evidence = agentic_context.strip() or related_doc
        messages = [
            SystemMessage(
                content=(
                    "你是一名校规问答助手。下面给你一段与用户问题相关的原文证据，"
                    "请严格基于这些证据回答问题，尽量引用原文表述，不要编造文档中没有的信息。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：{query}\n\n"
                    f"相关证据：\n{evidence}\n\n"
                    "请根据上述证据，用中文回答用户的问题。"
                )
            ),
        ]
        final_answer_msg = self.llm.invoke(messages)
        final_answer = getattr(final_answer_msg, "content", str(final_answer_msg))

        record = TrajectoryRecord(
            query=query,
            history=history,
            final_answer=final_answer,
            task_success=task_success,
            meta=meta,
        )
        self._append_record(record)

        return final_answer

