# -*- coding: utf-8 -*-
"""多租户注册表 + API Key 鉴权 + 角色权限（骨架，纯标准库，可脱离重库单测）。

半产品化思路：部署时只需编辑项目根目录的 tenants.json 定义租户，不用改代码。

    tenants.json 结构：
    {
      "tenants": {
        "<tenant_id>": {
          "name": "显示名",
          "api_key": "mk-xxx",            # 用 generate_api_key() 生成；或存 hash_key() 摘要
          "role": "admin" | "user",        # admin 可建索引/删会话，user 只能问答
          "kb": "<知识库id>",              # 可选，默认=tenant_id；多个租户可共享同一知识库
          "docs_dir": "tenants/<id>/docs"  # 可选，provision.py 建索引用
        }
      }
    }

快速生成一份可用配置：python -m rag.tenants
"""
import hashlib
import json
import logging
import os
import secrets

from .config import BASE_DIR, INDEX_DIR

logger = logging.getLogger("rag.tenants")

TENANTS_FILE = os.getenv("RAG_TENANTS_FILE", os.path.join(BASE_DIR, "tenants.json"))

ADMIN_ROLE = "admin"
USER_ROLE = "user"
ROLES = (ADMIN_ROLE, USER_ROLE)


class TenantError(PermissionError):
    """鉴权 / 权限相关错误，api 层统一转成 HTTPException。"""


def generate_api_key(prefix="mk-"):
    """生成随机 API Key（用 secrets，适合直接发给租户）。"""
    return prefix + secrets.token_urlsafe(24)


def hash_key(key):
    """把 key 摘要后存盘（可选加固，避免明文落盘）。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_tenants(path=None):
    """读 tenants.json → {tenant_id: tenant}。文件不存在或格式不对返回 {}。"""
    path = path or TENANTS_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("tenants.json 解析失败：%s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    tenants = data.get("tenants", data)
    return {k: v for k, v in tenants.items() if isinstance(v, dict)}


def _key_matches(candidate, stored):
    """常量时间比对，防时序侧信道；兼容明文 key 与 sha256 摘要两种存法。"""
    if not candidate or not stored:
        return False
    if secrets.compare_digest(candidate, stored):
        return True
    return secrets.compare_digest(hash_key(candidate), stored)


def get_tenant(api_key, tenants=None):
    """按 api_key 找租户，返回 (tenant_id, tenant) 或 (None, None)。"""
    tenants = load_tenants() if tenants is None else tenants
    if not api_key:
        return None, None
    for tid, tenant in tenants.items():
        if _key_matches(api_key, tenant.get("api_key", "")):
            return tid, tenant
    return None, None


def tenant_scope(tenant_id, tenant):
    """租户的存储 scope（知识库 id）。kb 缺省时等于 tenant_id。"""
    return tenant.get("kb") or tenant_id


def require_role(api_key, role, tenants=None):
    """校验 api_key 并要求指定角色，返回 (tenant_id, tenant)；否则抛 TenantError。"""
    tenants = load_tenants() if tenants is None else tenants
    if not tenants:
        raise TenantError("未配置 tenants.json，多租户模式未启用")
    tenant_id, tenant = get_tenant(api_key, tenants)
    if tenant_id is None:
        raise TenantError("无效的 API Key")
    if tenant.get("role", USER_ROLE) != role:
        raise TenantError(f"权限不足：需要 {role} 角色")
    return tenant_id, tenant


def index_dir_for(scope, base=None):
    """某个租户（scope）的索引目录：<INDEX_DIR>/<scope>。"""
    base = base or INDEX_DIR
    return os.path.join(base, scope)


if __name__ == "__main__":
    # 生成一份可用的 tenants.json 示例（含随机 key），直接重定向到文件即可：
    #   python -m rag.tenants > tenants.json
    sample = {
        "tenants": {
            "customer_admin": {
                "name": "客户 · 管理员",
                "api_key": generate_api_key(),
                "role": "admin",
                "kb": "customer",
                "docs_dir": "tenants/customer/docs",
            },
            "customer_user": {
                "name": "客户 · 员工",
                "api_key": generate_api_key(),
                "role": "user",
                "kb": "customer",
                "docs_dir": None,
            },
        }
    }
    print(json.dumps(sample, ensure_ascii=False, indent=2))
