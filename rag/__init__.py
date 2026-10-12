# -*- coding: utf-8 -*-
"""RAG 核心包：对外统一导出。

分层：
    config     配置（环境变量 / .env）
    loader     文档加载（txt / pdf / docx）
    chunker    切块
    embedder   向量化（bge）
    indexer    FAISS 索引（构建 + 持久化）
    retriever  检索
    generator  生成（DeepSeek）
    history    会话记忆（SQLite）
    pipeline   编排
"""
from .config import (
    API_KEY, BASE_URL, MODEL, TIMEOUT,
    EMBED_MODEL_NAME, HF_ENDPOINT,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, DOCS_DIR,
    INDEX_DIR, CHAT_HISTORY_DB, setup_logging,
)
from .chunker import chunk_text
from .loader import load_docs_folder, read_file, read_file_bytes
from .embedder import get_embedder, embed_texts
from .indexer import build_index, load_index, save_index
from .retriever import retrieve
from .generator import build_prompt, generate, generate_stream
from .history import add_message, get_history, clear_session
from .pipeline import build_from_folder, build_from_files, query

__all__ = [
    "API_KEY", "BASE_URL", "MODEL", "TIMEOUT",
    "EMBED_MODEL_NAME", "HF_ENDPOINT",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "DOCS_DIR",
    "INDEX_DIR", "CHAT_HISTORY_DB", "setup_logging",
    "chunk_text",
    "load_docs_folder", "read_file", "read_file_bytes",
    "get_embedder", "embed_texts",
    "build_index", "load_index", "save_index",
    "retrieve",
    "build_prompt", "generate", "generate_stream",
    "add_message", "get_history", "clear_session",
    "build_from_folder", "build_from_files", "query",
]
