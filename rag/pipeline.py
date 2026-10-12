# -*- coding: utf-8 -*-
"""RAG 流水线编排：把加载→切块→向量化→索引→检索→生成串起来。"""
import logging

from .config import DOCS_DIR, TOP_K
from .generator import generate, generate_stream
from .indexer import build_index, load_index, save_index
from .loader import load_docs_folder
from .retriever import retrieve

logger = logging.getLogger("rag.pipeline")


def build_from_folder(folder=None):
    """从文档文件夹建立索引，返回 (index, chunks)。"""
    folder = folder or DOCS_DIR
    docs = load_docs_folder(folder)
    if not docs:
        raise ValueError(f"文件夹里没有可读取的文档：{folder}")
    return build_index(docs)


def build_from_files(files):
    """从 [(文件名, 文本), ...] 建立索引（网页/API 上传用）。"""
    docs = [(name, text) for name, text in files if text and text.strip()]
    if not docs:
        raise ValueError("没有可入库的文档内容")
    return build_index(docs)


def query(index, chunks, question, top_k=TOP_K):
    """检索 + 生成（非流式），返回 (answer, sources)。sources 是 [(score, chunk), ...]。"""
    results = retrieve(index, chunks, question, top_k)
    answer = generate(question, results)
    return answer, results
