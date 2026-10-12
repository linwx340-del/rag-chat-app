# -*- coding: utf-8 -*-
"""history 会话记忆单测（stdlib sqlite3，不依赖重库）。"""
import os
import tempfile

import rag.history as history


def _tmp_db():
    tmp = tempfile.mkdtemp()
    return os.path.join(tmp, "test.db")


def test_add_and_get_history():
    history.CHAT_HISTORY_DB = _tmp_db()
    history.clear_session("s1")
    history.add_message("s1", "user", "你好")
    history.add_message("s1", "assistant", "你好，有什么可以帮你？")
    assert history.get_history("s1") == [
        ("user", "你好"),
        ("assistant", "你好，有什么可以帮你？"),
    ]


def test_history_isolated_by_session():
    history.CHAT_HISTORY_DB = _tmp_db()
    history.clear_session("s1")
    history.clear_session("s2")
    history.add_message("s1", "user", "问题A")
    history.add_message("s2", "user", "问题B")
    assert [c for _, c in history.get_history("s1")] == ["问题A"]
    assert [c for _, c in history.get_history("s2")] == ["问题B"]


def test_clear_session():
    history.CHAT_HISTORY_DB = _tmp_db()
    history.add_message("s1", "user", "x")
    history.clear_session("s1")
    assert history.get_history("s1") == []


def test_history_limit():
    history.CHAT_HISTORY_DB = _tmp_db()
    history.clear_session("s1")
    for i in range(10):
        history.add_message("s1", "user", f"消息{i}")
    rows = history.get_history("s1", limit=3)
    assert len(rows) == 3
    assert rows[-1][1] == "消息9"   # 保留最近 3 条，最后一条是最新的
