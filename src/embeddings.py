from typing import Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_PATH = "/data/home/xmju/models/Qwen3-Embedding-0.6B"


def get_embeddings(
    model_path: str = EMBEDDING_MODEL_PATH,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )
    return embeddings

