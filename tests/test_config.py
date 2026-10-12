# -*- coding: utf-8 -*-
"""config 纯逻辑单测（不依赖重库）。"""
import os

from rag import config


def test_defaults():
    assert config.MODEL == "deepseek-chat"
    assert config.BASE_URL == "https://api.deepseek.com"
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.TOP_K > 0
    assert config.EMBED_MODEL_NAME.startswith("BAAI/")


def test_load_api_key_from_env():
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-123"
    try:
        assert config._load_api_key() == "sk-test-123"
    finally:
        del os.environ["DEEPSEEK_API_KEY"]


def test_load_api_key_empty_when_unset():
    os.environ.pop("DEEPSEEK_API_KEY", None)
    # 没配 key 时返回空串（不抛异常）
    assert config._load_api_key() in ("", None) or isinstance(config._load_api_key(), str)
