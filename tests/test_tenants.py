# -*- coding: utf-8 -*-
"""多租户 / 鉴权 纯逻辑单测（标准库，不依赖重库）。"""
import json
import os
import tempfile

import rag.history as history
import rag.tenants as tenants


def _tmp_json(data):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tenants.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def test_generate_api_key():
    k1 = tenants.generate_api_key()
    k2 = tenants.generate_api_key()
    assert k1.startswith("mk-")
    assert len(k1) > 30
    assert k1 != k2


def test_hash_key_deterministic():
    h1 = tenants.hash_key("abc")
    h2 = tenants.hash_key("abc")
    assert h1 == h2
    assert len(h1) == 64
    assert h1 != "abc"


def test_load_tenants_missing():
    assert tenants.load_tenants("/nonexistent/path/to/tenants.json") == {}


def test_load_tenants_wrapped_and_flat():
    wrapped = {"tenants": {"a": {"name": "A", "api_key": "k1"}}}
    assert tenants.load_tenants(_tmp_json(wrapped))["a"]["name"] == "A"
    flat = {"b": {"name": "B", "api_key": "k2"}}
    assert tenants.load_tenants(_tmp_json(flat))["b"]["name"] == "B"


def test_get_tenant_by_key():
    t = {"a": {"api_key": "mk-aaa", "role": "admin"}}
    tid, tenant = tenants.get_tenant("mk-aaa", t)
    assert tid == "a"
    assert tenant["role"] == "admin"
    assert tenants.get_tenant("wrong", t) == (None, None)
    assert tenants.get_tenant("", t) == (None, None)


def test_get_tenant_accepts_hashed_key():
    raw = "mk-secret"
    t = {"a": {"api_key": tenants.hash_key(raw), "role": "user"}}
    tid, _ = tenants.get_tenant(raw, t)
    assert tid == "a"


def test_require_role():
    t = {"a": {"api_key": "mk-admin", "role": "admin"},
         "b": {"api_key": "mk-user", "role": "user"}}
    assert tenants.require_role("mk-admin", "admin", t)[0] == "a"
    for key in ("mk-user", "nokey"):
        try:
            tenants.require_role(key, "admin", t)
            assert False, "应抛 TenantError"
        except tenants.TenantError:
            pass


def test_tenant_scope_fallback():
    assert tenants.tenant_scope("x", {}) == "x"
    assert tenants.tenant_scope("x", {"kb": "shared"}) == "shared"


def test_index_dir_for():
    assert tenants.index_dir_for("machinery", "/base") == os.path.join("/base", "machinery")


def test_history_isolated_by_tenant():
    db = os.path.join(tempfile.mkdtemp(), "t.db")
    history.CHAT_HISTORY_DB = db
    history.add_message("s1", "user", "甲的问题", tenant_id="t1")
    history.add_message("s1", "user", "乙的问题", tenant_id="t2")
    assert [c for _, c in history.get_history("s1", tenant_id="t1")] == ["甲的问题"]
    assert [c for _, c in history.get_history("s1", tenant_id="t2")] == ["乙的问题"]


def test_history_backward_compat_no_tenant():
    db = os.path.join(tempfile.mkdtemp(), "t.db")
    history.CHAT_HISTORY_DB = db
    history.add_message("s1", "user", "旧数据")
    assert [c for _, c in history.get_history("s1")] == ["旧数据"]
