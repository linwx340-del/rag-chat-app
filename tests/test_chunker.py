# -*- coding: utf-8 -*-
"""chunker 纯逻辑单测（不依赖重库）。"""
from rag.chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_keeps_sentence_punctuation():
    # 短文本默认合并成一块，句号被保留
    assert chunk_text("你好。世界。") == ["你好。世界。"]
    # 强制切分时，每块仍保留句号
    assert chunk_text("你好。世界。", chunk_size=3, overlap=0) == ["你好。", "世界。"]


def test_single_short_sentence():
    assert chunk_text("一句话。") == ["一句话。"]


def test_long_text_splits_into_multiple():
    text = "一二三四五六七八九十。" * 20   # 共 220 字
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 4


def test_no_empty_chunks():
    text = "一二三四五六七八九十。" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert all(c.strip() for c in chunks)


def test_no_content_lost_at_ends():
    text = "一二三四五六七八九十。" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    joined = "".join(chunks)
    assert text[:20] in joined    # 开头没丢
    assert text[-20:] in joined   # 结尾没丢


def test_zero_chunk_size_returns_whole_text():
    text = "你好。世界。再见。"
    assert chunk_text(text, chunk_size=0) == [text]
