# -*- coding: utf-8 -*-
"""会话记忆：SQLite 按 (tenant_id, session_id) 存对话历史，跨请求 / 重启持久化。

纯标准库（sqlite3），可脱离重库单测。
多租户：每个租户的会话彼此隔离；tenant_id 缺省为 ""（单租户 / 老数据兼容）。
"""
import os
import sqlite3
import threading

from .config import CHAT_HISTORY_DB

_lock = threading.Lock()


def _tid(tenant_id):
    """租户 id 归一化：None → ""（单租户模式 / 老数据兼容）。"""
    return tenant_id if tenant_id is not None else ""


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
    # 老库没有 tenant_id 列：迁移补上，默认空串（向后兼容单租户数据）
    cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if "tenant_id" not in cols:
        conn.execute(
            "ALTER TABLE messages ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_tenant_session "
        "ON messages (tenant_id, session_id)")
    conn.commit()
    return conn


def add_message(session_id, role, content, tenant_id=None):
    with _lock:
        conn = _conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tenant_id) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _tid(tenant_id)),
        )
        conn.commit()
        conn.close()


def get_history(session_id, limit=6, tenant_id=None):
    """返回 [(role, content), ...]，按时间正序（最近的 limit 条）。"""
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages "
        "WHERE session_id = ? AND tenant_id = ? "
        "ORDER BY id DESC LIMIT ?",
        (session_id, _tid(tenant_id), limit),
    ).fetchall()
    conn.close()
    return list(reversed(rows))


def clear_session(session_id, tenant_id=None):
    with _lock:
        conn = _conn()
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND tenant_id = ?",
            (session_id, _tid(tenant_id)),
        )
        conn.commit()
        conn.close()
