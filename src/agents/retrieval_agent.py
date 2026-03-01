from __future__ import annotations

from dataclasses import dataclass

from ..tools.retrieval import retrieval_augment


@dataclass
class RetrievalAgent:
    """
    检索增强 Agent：封装 retrieval_augment tool，便于在非 Agent 场景下直接调用。
    """

    def run(self, query: str, keyword: str) -> str:
        return retrieval_augment.invoke({"query": query, "keyword": keyword})

