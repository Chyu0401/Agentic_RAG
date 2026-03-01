from __future__ import annotations

from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class SummaryRelatedDocFormat(BaseModel):
    """对用户 query 相关的片段进行筛选结果。"""

    summary_related_doc_res: str = Field(
        ...,
        description="仅包含与用户 query 强相关的原文片段的汇总，不得改写原文，只做摘取与拼接。",
    )


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个乐于助人的笔记精读助手。"
            "现在有很多原文片段（related_doc），以及用户的 query。"
            "你的任务是：只从这些原文片段中**摘取与 query 高度相关的句子或段落**，"
            "不要改写原文内容，不要新增解释，只做复制-筛选-拼接。",
        ),
        (
            "human",
            "这是用户的 query：\n{query}\n\n"
            "这是原文的众多片段（related_doc）：\n{related_doc}",
        ),
    ]
)


def summary_related_doc(
    llm: BaseChatModel,
    query: str,
    related_doc: str,
) -> SummaryRelatedDocFormat:
    """
    片段精读：根据 query，在 related_doc 中筛选出相关原文片段（不改写，只复制）。
    """
    structured_llm = llm.with_structured_output(SummaryRelatedDocFormat)
    chain = _SUMMARY_PROMPT | structured_llm
    return chain.invoke({"query": query, "related_doc": related_doc})


def make_summary_related_doc_tool(llm: BaseChatModel):
    """
    构造一个可用于 Agent 的精读工具。
    通过闭包注入 llm，避免使用全局 model 变量。
    """

    @tool(
        name="summary_related_doc",
        description=(
            "片段精读。根据用户 query，在 related_doc 原始片段中筛选出与 query 相关的原文，"
            "不进行改写，仅复制相关部分。"
        ),
    )
    def _summary_related_doc(query: str, related_doc: str) -> SummaryRelatedDocFormat:
        return summary_related_doc(llm=llm, query=query, related_doc=related_doc)

    return _summary_related_doc

