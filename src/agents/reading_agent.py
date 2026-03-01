from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..tools.reading import SummaryRelatedDocFormat, summary_related_doc


@dataclass
class ReadingAgent:
    """
    精读 Agent：对检索出来的 related_doc 做证据级筛选，只保留与 query 强相关的原文片段。
    """

    llm: BaseChatModel

    def run(self, query: str, related_doc: str) -> SummaryRelatedDocFormat:
        return summary_related_doc(llm=self.llm, query=query, related_doc=related_doc)

