# -*- coding: utf-8 -*-
"""半产品化初始化：读 tenants.json，为每个配置了 docs_dir 的租户建索引并落盘。

用法（在 learn_ai 目录下）：
    python provision.py            # 处理所有租户
    python provision.py <tid>      # 只处理某个租户

依赖：sentence-transformers + faiss（建索引需要，见 requirements.txt）。
多租户共享知识库：多个租户配同一个 kb 时，只需其中一个配 docs_dir，索引只建一份。
"""
import argparse
import os
import sys

from rag import build_from_folder, save_index, setup_logging
from rag.config import BASE_DIR
from rag.tenants import index_dir_for, load_tenants, tenant_scope

setup_logging()


def provision(tenant_id=None):
    tenants = load_tenants()
    if not tenants:
        print("未找到 tenants.json（或为空）。")
        print("  快速开始：python -m rag.tenants > tenants.json  # 生成一份含随机 key 的配置")
        print("  然后编辑 tenants.json 填 docs_dir，再把文档放进对应目录。")
        return 1
    if tenant_id and tenant_id not in tenants:
        print(f"租户 {tenant_id!r} 不存在。现有租户：{', '.join(tenants)}")
        return 1

    targets = {tenant_id: tenants[tenant_id]} if tenant_id else tenants
    for tid, t in targets.items():
        docs_dir = t.get("docs_dir")
        if not docs_dir:
            print(f"[跳过] {tid}：未配置 docs_dir（可能与其他租户共享同一 kb）")
            continue
        if not os.path.isabs(docs_dir):
            docs_dir = os.path.join(BASE_DIR, docs_dir)
        scope = tenant_scope(tid, t)
        print(f"[建索引] {tid} (scope={scope}) ← {docs_dir}")
        try:
            index, chunks = build_from_folder(docs_dir)
            save_index(index, chunks, index_dir_for(scope))
        except Exception as e:
            print(f"[失败] {tid}: {e}")
            return 1
        print(f"[完成] {tid}: 入库 {len(chunks)} 个片段")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="按 tenants.json 为各租户建索引")
    p.add_argument("tenant_id", nargs="?", default=None, help="只处理指定租户")
    args = p.parse_args()
    sys.exit(provision(args.tenant_id))
