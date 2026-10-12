# -*- coding: utf-8 -*-
"""RAG 命令行版——核心逻辑在 rag 包。

用法：
    1. 复制 .env.example 为 .env，填 DEEPSEEK_API_KEY
    2. 把 .txt / .pdf / .docx 放进 docs/ 文件夹
    3. python cli.py
"""
from rag import (API_KEY, DOCS_DIR, TOP_K, setup_logging,
                 build_from_folder, retrieve, generate)


def main():
    setup_logging()
    if not API_KEY:
        print("未设置 DeepSeek API Key。请复制 .env.example 为 .env 并填入 key，")
        print("或设置环境变量 DEEPSEEK_API_KEY。")
        return

    try:
        index, chunks = build_from_folder(DOCS_DIR)
    except (ValueError, ImportError) as e:
        print(f"建索引失败：{e}")
        return
    print(f"已入库 {len(chunks)} 个文本块\n")

    while True:
        query = input("请输入问题（输入 quit 退出）：").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = retrieve(index, chunks, query, TOP_K)
        print(f"\n检索到 {len(results)} 块，相似度：", [round(s, 4) for s, _ in results])
        answer = generate(query, results)
        print(f"\n回答：{answer}\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
