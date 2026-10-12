# -*- coding: utf-8 -*-
"""会话记忆：SQLite 按 session_id 存对话历史，跨请求 / 重启持久化。

纯标准库（sqlite3），可脱离重库单测。
"""
import os
import sqlite3
import threading

from .config import CHAT_HISTORY_DB

_lock = threading.Lock()


def _conn():
    db_dir = os.path.dirname(CHAT_HISTORY_DB)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(CHAT_HISTORY_DB, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "session_id TEXT, role TEXT, content TEXT,"
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()
    return conn


def add_message(session_id, role, content):
    with _lock:
        conn = _conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        conn.close()


def get_history(session_id, limit=6):
    """返回 [(role, content), ...]，按时间正序（最近的 limit 条）。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? "
        "ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return list(reversed(rows))


def clear_session(session_id):
    with _lock:
        conn = _conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
