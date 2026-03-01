import argparse
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore import create_vectorstore_from_documents


def build_index(
    data_dir: str,
    globs: List[str],
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    docs = []
    for g in globs:
        loader = DirectoryLoader(
            str(data_path),
            glob=g,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        docs.extend(loader.load())

    if not docs:
        raise RuntimeError(f"未从 {data_dir} 加载到任何文件（globs={globs}）。")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} files, split into {len(documents)} chunks.")

    create_vectorstore_from_documents(
        documents=documents,
        persist_directory=persist_dir,
    )
    print(f"Chroma index persisted to: {persist_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Chroma 向量数据库索引")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/internal_docs",
        help="原始语料目录",
    )
    parser.add_argument(
        "--globs",
        nargs="+",
        default=["**/*.txt", "**/*.md"],
        help="加载文件匹配模式（可传多个），例如：--globs **/*.txt **/*.md",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./rag_cache/chroma_db",
        help="Chroma 持久化目录（与检索时保持一致）",
    )
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        data_dir=args.data_dir,
        globs=args.globs,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

