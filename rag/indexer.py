# -*- coding: utf-8 -*-
"""FAISS 索引：构建 + 持久化（存磁盘免重建）。"""
import json
import logging
import os

from .chunker import chunk_text
from .embedder import embed_texts

logger = logging.getLogger("rag.indexer")


def build_index(docs):
    """docs: [(文件名, 文本), ...] → 返回 (index, chunks)。

    chunks 每个元素是 {"file": 文件名, "text": 片段}。
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("缺少 faiss，请先运行：pip install faiss-cpu")

    chunks = []
    for fname, text in docs:
        for c in chunk_text(text):
            if c.strip():
                chunks.append({"file": fname, "text": c})
    if not chunks:
        raise ValueError("没有可入库的文本块，请检查文档内容是否为文字版")

    vectors = embed_texts([c["text"] for c in chunks])
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP = 内积，配合归一化向量做余弦相似度
    index.add(vectors)
    logger.info("已入库 %d 个文本块，向量维度 %d", len(chunks), dim)
    return index, chunks


def save_index(index, chunks, index_dir):
    """把 FAISS 索引和 chunks 存到磁盘，下次免重建。"""
    import faiss
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("索引已保存到 %s", index_dir)


def load_index(index_dir):
    """从磁盘读回 (index, chunks)；不存在则返回 (None, None)。"""
    import faiss
    idx_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.json")
    if not (os.path.exists(idx_path) and os.path.exists(chunks_path)):
        return None, None
    index = faiss.read_index(idx_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("从磁盘加载索引：%d 个文本块", len(chunks))
    return index, chunks
