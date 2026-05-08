"""VeRL multi-turn 训练用 RetrievalTool。

Pure ReAct 模式下，模型唯一可调用的工具。包装 Chroma 向量检索 + 关键词重排，
通过 BaseTool 异步接口暴露给 VeRL multi-turn rollout。

工具参数（OpenAI function 格式，定义见 tool_config/rag_tools.yaml）：
- query (str)：模型自己改写后的检索语句
- keyword (str)：模型从 query 抽出的核心关键词，用于命中加权（可选）

返回：top-K 文档片段拼接为单段文本，供模型在下一轮 thinking 中继续推理。
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# 进程级单例 vectorstore（每个 Ray rollout worker 加载一次）
# 避免每条 trajectory 重建 Chroma client + 重新加载 embedding 模型
_VECTORSTORE = None
_VECTORSTORE_LOCK = asyncio.Lock()


def _load_vectorstore(persist_directory: str):
    """同步加载 Chroma vectorstore。延迟 import，避免在 Ray import 阶段报错。"""
    from langchain_chroma import Chroma

    # 延迟 import 项目内 embeddings 模块（依赖 PYTHONPATH 包含项目根目录）
    from src.embeddings import get_embeddings

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )


async def _get_vectorstore(persist_directory: str):
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    async with _VECTORSTORE_LOCK:
        if _VECTORSTORE is None:
            _VECTORSTORE = await asyncio.to_thread(_load_vectorstore, persist_directory)
            logger.info(f"Loaded Chroma vectorstore from {persist_directory}")
    return _VECTORSTORE


def _keyword_reweight(docs_with_scores, keyword: str, bonus: float = 0.1):
    """命中关键词的 chunk，距离减 bonus（Chroma 距离越小越相近）。"""
    if not keyword:
        return docs_with_scores
    adjusted = []
    for doc, score in docs_with_scores:
        if keyword in (doc.page_content or ""):
            score = score - bonus
        adjusted.append((doc, score))
    adjusted.sort(key=lambda x: x[1])
    return adjusted


class RetrievalTool(BaseTool):
    """RAG 检索工具。

    config 字段：
        persist_directory (str): Chroma 向量库目录（相对项目根目录或绝对路径）
        topk_recall (int): 初步召回数量（default 10）
        topk_return (int): 重排后返回给模型的数量（default 5）
        keyword_bonus (float): 命中关键词时距离减分（default 0.1）
        max_chars_per_chunk (int): 单条 chunk 截断长度，避免 KV cache 爆炸（default 800）
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.persist_directory = config.get("persist_directory", "./rag_cache/chroma_db")
        self.topk_recall = int(config.get("topk_recall", 10))
        self.topk_return = int(config.get("topk_return", 5))
        self.keyword_bonus = float(config.get("keyword_bonus", 0.1))
        self.max_chars_per_chunk = int(config.get("max_chars_per_chunk", 800))
        self._instance_dict: dict[str, dict] = {}
        logger.info(
            f"Initialized RetrievalTool: persist={self.persist_directory}, "
            f"topk_recall={self.topk_recall}, topk_return={self.topk_return}"
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "call_count": 0,
            "results_text": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        keyword = parameters.get("keyword", "") or ""
        if not isinstance(query, str):
            query = str(query)
        if not isinstance(keyword, str):
            keyword = str(keyword)

        if not query.strip():
            error_text = "Error: 'query' is empty. Please provide a non-empty search query."
            return ToolResponse(text=error_text), 0.0, {"error": "empty_query"}

        try:
            vectorstore = await _get_vectorstore(self.persist_directory)

            # Chroma similarity_search_with_score 是同步阻塞 IO，丢到线程池
            docs_with_scores = await asyncio.to_thread(
                vectorstore.similarity_search_with_score, query, self.topk_recall
            )

            docs_with_scores = _keyword_reweight(docs_with_scores, keyword, self.keyword_bonus)
            top_docs = [doc for doc, _ in docs_with_scores[: self.topk_return]]

            # 拼接前对每条 chunk 截断，防止单次返回吃光 KV cache
            chunks = []
            for i, doc in enumerate(top_docs):
                content = (doc.page_content or "").strip()
                if not content:
                    continue
                if len(content) > self.max_chars_per_chunk:
                    content = content[: self.max_chars_per_chunk] + "..."
                chunks.append(f"[doc {i + 1}] {content}")

            if not chunks:
                result_text = "No relevant documents found."
            else:
                result_text = "\n\n".join(chunks)

            # 记录到 instance state，供 reward 函数读取（如需）
            self._instance_dict[instance_id]["call_count"] += 1
            self._instance_dict[instance_id]["results_text"].append(result_text)

            metrics = {
                "n_docs_returned": len(chunks),
                "query_len": len(query),
                "has_keyword": bool(keyword.strip()),
                "call_count": self._instance_dict[instance_id]["call_count"],
            }
            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            logger.error(f"[RetrievalTool] Execution failed: {e}", exc_info=True)
            return (
                ToolResponse(text=f"Retrieval error: {e}"),
                0.0,
                {"error": str(e)},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # 工具本身不直接给 step reward，全部由全局 compute_score 在轨迹结束时算
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
