from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..tools.query_preprocess import ExpandAndKeywordFormat, expand_and_keyword


@dataclass
class QueryPreprocessAgent:
    """
    基础版 Query 预处理 Agent（不走 ReAct 多步循环）。

    后续接入完整 Agentic RAG 时，可把 expand_and_keyword 作为一个 tool
    交给 ReAct Agent 调用；这里先提供一个直接调用的轻量封装，便于先跑通流程。
    """

    llm: BaseChatModel

    def run(self, query: str) -> ExpandAndKeywordFormat:
        return expand_and_keyword(llm=self.llm, query=query)

