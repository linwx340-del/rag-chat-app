# -*- coding: utf-8 -*-
"""集中配置：所有可调参数都从这里读。

读取优先级（高 → 低）：
    1. 环境变量（DEEPSEEK_API_KEY / RAG_* 等）
    2. 项目根目录 .env 文件（python-dotenv 自动加载）
    3. key.txt 兜底（老方式，已 gitignore）

新手最快上手：复制 .env.example 为 .env，填上 DEEPSEEK_API_KEY 即可。
"""
import os
import logging

# 项目根目录（learn_ai/），key.txt / .env / docs 都相对这里
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 自动加载项目根目录的 .env（用 python-dotenv）。没装也不崩：跳过即可。
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass


def _load_api_key() -> str:
    """按优先级读 key：环境变量 > .env（load_dotenv 已注入）> key.txt 兜底。"""
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    key_file = os.path.join(BASE_DIR, "key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
    return key


# ---- 生成（DeepSeek，OpenAI 兼容接口）----
API_KEY = _load_api_key()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "60"))

# ---- 向量化 / 检索 ----
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")  # 国内下载 bge 模型加速

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
DOCS_DIR = os.getenv("RAG_DOCS_DIR", os.path.join(BASE_DIR, "docs"))

# ---- 持久化（阶段4：索引落盘 + 会话记忆）----
INDEX_DIR = os.getenv("RAG_INDEX_DIR", os.path.join(BASE_DIR, ".rag_index"))
CHAT_HISTORY_DB = os.getenv("CHAT_HISTORY_DB", os.path.join(BASE_DIR, "chat_history.db"))


def setup_logging(level: int = logging.INFO) -> None:
    """统一日志格式，便于线上排查问题。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
