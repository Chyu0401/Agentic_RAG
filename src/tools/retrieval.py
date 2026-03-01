from __future__ import annotations

from typing import List, Tuple

from langchain.tools import tool
from langchain_core.documents import Document

from ..vectorstore import get_vectorstore


def _keyword_reweight_docs(
    docs_with_scores: List[Tuple[Document, float]],
    keyword: str,
    bonus: float = 0.1,
) -> List[Tuple[Document, float]]:
    """
    在向量检索结果的基础上进行关键词加权：
    - Chroma 使用距离（越小越相近），因此命中关键词时将距离减去 bonus。
    """
    if not keyword:
        return docs_with_scores

    adjusted: List[Tuple[Document, float]] = []
    for doc, score in docs_with_scores:
        if keyword in (doc.page_content or ""):
            score = score - bonus
        adjusted.append((doc, score))

    # 按“距离”升序排序（越小越相似）
    adjusted.sort(key=lambda x: x[1])
    return adjusted


@tool
def retrieval_augment(query: str, keyword: str) -> str:
    """
    【RAG 检索增强】基于 query 和核心关键词从 Chroma 向量库中检索相关文档，并对结果进行关键词感知的重排。

    流程：
    1. 使用 query 进行向量检索，取 Top10（带距离分数，距离越小越相似）。
    2. 若提供 keyword，则在检索到的文本中进行关键词匹配：
       - 若某个 chunk 包含 keyword，则在其原有“距离分数”的基础上减去 0.1，使其更靠前。
    3. 对加权后的结果按距离升序排序，取 Top5。
    4. 将 Top5 的文本内容用两个换行符拼接返回。

    返回：
        related_doc: str，按优化后得分排序的文档内容，以双换行符分隔拼接。
    """
    vectorstore = get_vectorstore()

    # 1. 基于 query 做向量检索（带分数）
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)

    # 2. 如果有关键词，对包含关键词的 chunk 进行“距离减 0.1”的加权
    docs_with_scores = _keyword_reweight_docs(
        docs_with_scores=docs_with_scores,
        keyword=keyword,
        bonus=0.1,
    )

    # 3. 取前 5 个片段
    top_docs = [doc for doc, _ in docs_with_scores[:5]]

    # 4. 拼接为一个长上下文
    related_doc = "\n\n".join(doc.page_content for doc in top_docs if doc.page_content)
    return related_doc

