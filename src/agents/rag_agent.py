from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ..tools.query_preprocess import make_expand_and_keyword_tool
from ..tools.reading import make_summary_related_doc_tool
from ..tools.retrieval import retrieval_augment


class AgentResult(BaseModel):
    """对用户 query 进行最终回答。"""

    agent_answer: str = Field(..., description="对用户 query 的回答，基于检索到的证据。")


SYSTEM_PROMPT = (
    "你是一名助人为乐的助手，你会根据用户的问题进行判断：\n\n"
    "- 如果用户问题与校规不相关，那么就直接进行回答，不要调用工具。\n\n"
    "- 如果用户问题与校规相关，就需要进行 RAG 检索，然后根据检索到的相关文档进行回答，具体步骤如下：\n"
    "  1. 可选择：如果用户描述不清楚，你需要调用 expand_and_keyword 函数对用户问题进行改写并提取关键词。\n"
    "  2. 必须进行：调用 retrieval_augment 函数，基于用户输入的查询语句和核心关键词，从知识库中检索相关文档。\n"
    "  3. 可选择：若 retrieval_augment 检索后的文章大于 1000 字，可调用 summary_related_doc 函数"
    " 对检索的文档进行精读，抽取筛选与 query 相关的字段。\n"
    "  4. 根据精读的内容对用户的原始 query 进行回答；若没有找到答案，请再次对问题进行改写和关键词提取，"
    "然后再次进行 RAG 检索（最多重复两次）。\n"
    "  5. 若重复两次还没找到答案，请回答：没有在文档中检索到相关内容，所以无法准确回答。\n"
)


@dataclass
class RagAgent:
    """
    顶层 RAG Agent：基于 langchain.agents.create_agent，
    串联 query 预处理 / 检索增强 / 精读三个工具，完成多步 RAG 问答。
    """

    llm: BaseChatModel

    def __post_init__(self) -> None:
        expand_and_keyword_tool = make_expand_and_keyword_tool(self.llm)
        summary_related_doc_tool = make_summary_related_doc_tool(self.llm)

        tools = [
            expand_and_keyword_tool,
            retrieval_augment,
            summary_related_doc_tool,
        ]

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=AgentResult,
        )

    def run(self, query: str) -> AgentResult:
        """
        执行完整的 Agentic RAG 流程，对用户 query 给出回答。
        """
        result = self._agent.invoke({"messages": [("user", query)]})
        # create_agent + response_format 会直接返回 AgentResult 实例
        return result

