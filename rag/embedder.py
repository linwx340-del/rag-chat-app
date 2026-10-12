# -*- coding: utf-8 -*-
"""向量化：sentence-transformers + bge 模型，懒加载 + 进程内单例。

模型加载很慢（首次还要下载），所以做成单例：同一进程里只加载一次。
"""
import logging
import os

from .config import EMBED_MODEL_NAME, HF_ENDPOINT

logger = logging.getLogger("rag.embedder")

_embedder = None


def get_embedder():
    """懒加载 bge 向量模型（进程内单例）。"""
    global _embedder
    if _embedder is None:
        os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "缺少 sentence-transformers，请先运行：pip install sentence-transformers"
            )
        logger.info("正在加载向量模型 %s（首次会下载，请耐心等待）", EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed_texts(texts):
    """批量向量化；归一化之后，向量内积就等于余弦相似度。"""
    if not texts:
        raise ValueError("texts 不能为空")
    return get_embedder().encode(list(texts), normalize_embeddings=True)
