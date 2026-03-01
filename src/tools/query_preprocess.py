from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ExpandAndKeywordFormat(BaseModel):
    """对用户 query 进行改写/扩写与关键词提取的结果。"""

    expand_query: str = Field(
        ...,
        description="对 query 进行扩写/改写后的结果；如无需改写，可返回原 query。",
    )
    keyword: str = Field(
        ...,
        description="从 query 中提取的最重要的一个简短关键词（尽量短、具体）。",
    )


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个知识库问答系统的查询预处理助手。"
            "你的任务：\n"
            "1) 对用户 query 进行必要的改写/扩写，使其更完整、更适合检索（如果不需要改写，就保持原意并尽量保持原句）。\n"
            "2) 从 query 中提取一个最重要的“简短关键词”，用于后续对候选 chunk 的关键词匹配加权。\n"
            "要求：只输出结构化字段，不要输出多余解释。",
        ),
        ("human", "{query}"),
    ]
)


def expand_and_keyword(llm: BaseChatModel, query: str) -> ExpandAndKeywordFormat:
    """
    调用 LLM，返回结构化的改写 query 与关键词。
    """
    structured_llm = llm.with_structured_output(ExpandAndKeywordFormat)
    chain = _PROMPT | structured_llm
    return chain.invoke({"query": query})


def make_expand_and_keyword_tool(llm: BaseChatModel) -> StructuredTool:
    """
    构造一个可用于 Agent 的工具（tool）。

    通过闭包注入 llm，避免在工具内部依赖全局变量 model。
    """

    def _run(query: str) -> ExpandAndKeywordFormat:
        return expand_and_keyword(llm=llm, query=query)

    return StructuredTool.from_function(
        func=_run,
        name="expand_and_keyword",
        description="对用户输入的 query 进行改写/扩写，并提取一个最重要的简短关键词。",
    )

