from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import get_embeddings


CHROMA_PERSIST_DIR = "./rag_cache/chroma_db"


def get_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    连接到已有的 Chroma 向量库。
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore


def create_vectorstore_from_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    从一批文本构建 Chroma 向量库，并持久化到磁盘。
    """
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas 的长度必须与 texts 一致，或者直接传 None。")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
    )
    return vectorstore


def create_vectorstore_from_documents(
    documents: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    从 LangChain 的 Document 列表创建 Chroma 向量库。
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore

