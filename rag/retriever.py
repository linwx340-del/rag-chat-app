# -*- coding: utf-8 -*-
"""检索：把问题向量化，在 FAISS 索引里找 top_k 个最相关块。"""
import logging

from .config import TOP_K
from .embedder import embed_texts

logger = logging.getLogger("rag.retriever")


def retrieve(index, chunks, query, top_k=TOP_K):
    """返回 [(相似度, chunk), ...]，chunk 是 {"file", "text"}。"""
    q_vec = embed_texts([query])
    scores, ids = index.search(q_vec, top_k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i >= 0:   # -1 表示没检索到
            results.append((float(score), chunks[i]))
    logger.info("检索到 %d 个相关块", len(results))
    return results
