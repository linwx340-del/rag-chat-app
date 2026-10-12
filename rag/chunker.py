# -*- coding: utf-8 -*-
"""文本切块：按句子把长文本切成约 chunk_size 字的小块，相邻块重叠 overlap 字。

纯逻辑，不依赖任何重库，可单独单测。
"""
import re

from .config import CHUNK_SIZE, CHUNK_OVERLAP

_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按句子切块，相邻块保留 overlap 个字的上下文，避免一个意思被拦腰切断。

    chunk_size / overlap 为 0 或负数时，退化成"整篇一块"，不做切分。
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        return [text.strip()]

    sentences = _SENT_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    overlap = max(0, overlap)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > chunk_size:
            chunks.append(cur)
            # 注意：overlap=0 时 cur[-0:] 会取整串（Python 里 -0 == 0），
            # 所以显式判空，避免"零重叠"退化成"整块重叠"。
            cur = (cur[-overlap:] if overlap > 0 else "") + s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    return chunks
