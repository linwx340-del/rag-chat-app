# -*- coding: utf-8 -*-
"""loader 纯逻辑单测（只测 txt 读取，不碰 pdfplumber/docx）。"""
import os
import tempfile

from rag.loader import read_file, read_txt


def test_read_txt():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("你好，RAG。")
        path = f.name
    try:
        assert read_txt(path) == "你好，RAG。"
    finally:
        os.remove(path)


def test_read_file_dispatches_by_extension():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("内容")
        path = f.name
    try:
        assert read_file(path) == "内容"
    finally:
        os.remove(path)


def test_read_file_bytes_txt():
    from rag.loader import read_file_bytes
    assert read_file_bytes("a.txt", "hello".encode("utf-8")) == "hello"


def test_read_file_bytes_unknown_returns_empty():
    from rag.loader import read_file_bytes
    assert read_file_bytes("a.xyz", b"abc") == ""
