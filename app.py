# -*- coding: utf-8 -*-
"""RAG 网页版（Streamlit）—— 完整管理界面（面向交付）。

侧边栏：API Key 设置 + 文档管理（上传/保存/列表/删除/重建索引）+ 检索参数
主区：  数据看板（文档数/片段数/问答轮数/TOP_K）+ 问答区（流式 + 来源 + 多轮记忆）

持久化：上传的文档落盘到 docs/，索引落盘到 .rag_index/，刷新/重启免重建。
运行（在 learn_ai 目录下）：
    streamlit run app.py
"""
import os

import streamlit as st

from rag import (
    API_KEY, EMBED_MODEL_NAME, DOCS_DIR, INDEX_DIR,
    build_from_folder, load_index, save_index, retrieve, generate_stream,
)

st.set_page_config(page_title="RAG 知识问答", page_icon="📚", layout="wide")
st.title("📚 RAG 知识问答系统")
st.caption("上传文档 → 建立语义索引 → 提问，答案会标注它依据的原文片段")


# ================= 工具函数 =================
def list_docs(docs_dir):
    """列出 docs 目录下所有可读文档，返回 [(文件名, 字节数), ...]。"""
    if not os.path.isdir(docs_dir):
        return []
    files = []
    for name in sorted(os.listdir(docs_dir)):
        if name.lower().endswith((".txt", ".pdf", ".docx")):
            files.append((name, os.path.getsize(os.path.join(docs_dir, name))))
    return files


def save_uploaded(uploaded_files, docs_dir):
    """把上传的文件字节写入 docs_dir，返回成功保存的数量。"""
    os.makedirs(docs_dir, exist_ok=True)
    saved = 0
    for f in uploaded_files:
        path = os.path.join(docs_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getvalue())
        saved += 1
    return saved


def render_dashboard(doc_count, chunk_count, round_count, top_k):
    """顶部数据看板：四个统计卡片。"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 文档", f"{doc_count} 个")
    c2.metric("🧩 文本片段", f"{chunk_count} 个")
    c3.metric("💬 问答轮数", f"{round_count}")
    c4.metric("🎯 检索 TOP_K", f"{top_k}")


def render_sources(results):
    """来源引用：来源文件 + 相似度 + 原文片段（纯文本，避免前端组件 bug）。"""
    if not results:
        st.caption("未检索到相关片段")
        return
    for idx, (score, c) in enumerate(results, 1):
        st.markdown(f"**片段 {idx}** · 来自《{c['file']}》 · 相似度 `{score:.3f}`")
        st.write(c["text"])
        st.markdown("---")


# ================= 会话状态：首次运行时从磁盘加载已有索引（免重建） =================
if "index" not in st.session_state:
    st.session_state["index"] = None
    st.session_state["chunks"] = []
    st.session_state["messages"] = []
    st.session_state["del_note"] = ""
    try:
        _idx, _chunks = load_index(INDEX_DIR)
        if _idx is not None:
            st.session_state["index"] = _idx
            st.session_state["chunks"] = _chunks
    except Exception:
        pass  # 磁盘还没索引，或缺少 faiss，等上传后重建


# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 设置")
    secret_key = ""
    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        pass
    if secret_key:
        api_key = secret_key
        st.success("✅ 已使用云端配置的 API Key")
    elif API_KEY:
        api_key = API_KEY
        st.success("✅ 已读取本地 API Key（.env / 环境变量）")
    else:
        api_key = st.text_input("DeepSeek API Key", type="password",
                                placeholder="sk-...（本地运行填这里）")

    top_k = st.slider("检索片段数 TOP_K", 1, 8, 4)
    st.markdown("---")

    st.header("📁 文档管理")

    # 上传 + 保存（先于列表，保存后列表自然刷新，无需 rerun）
    uploaded = st.file_uploader(
        "上传文档（txt / pdf / docx，可多选）",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if st.button("💾 保存文档", disabled=not uploaded):
        n = save_uploaded(uploaded, DOCS_DIR)
        st.success(f"✅ 已保存 {n} 个文档，点下方「重建索引」生效")

    # 列出文档（在保存逻辑之后，保存后立刻能看到新文档）
    docs = list_docs(DOCS_DIR)

    if docs:
        st.caption(f"知识库里现有 {len(docs)} 个文档：")
        for name, size in docs:
            col_doc, col_del = st.columns([5, 1])
            col_doc.write(f"📄 {name}  `{size // 1024} KB`")
            if col_del.button("🗑", key=f"del_{name}", help=f"删除 {name}"):
                try:
                    os.remove(os.path.join(DOCS_DIR, name))
                    st.session_state["del_note"] = f"已删除 {name}，记得点「重建索引」"
                except OSError as e:
                    st.session_state["del_note"] = f"删除失败：{e}"
        if st.session_state["del_note"]:
            st.warning(st.session_state["del_note"])
            st.session_state["del_note"] = ""
    else:
        st.caption("知识库还是空的，上传文档后点「保存文档」")

    if docs:
        if st.button("🔨 重建索引", type="primary"):
            with st.spinner("正在读取全部文档、建立语义索引……"):
                try:
                    index, chunks = build_from_folder(DOCS_DIR)
                    save_index(index, chunks, INDEX_DIR)
                except ImportError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"建索引失败：{e}")
                else:
                    st.session_state["index"] = index
                    st.session_state["chunks"] = chunks
                    st.session_state["messages"] = []
                    st.success(f"✅ 索引已更新：{len(chunks)} 个文本片段，已落盘")

    st.markdown("---")
    st.caption(f"向量模型：`{EMBED_MODEL_NAME}`")
    st.caption("首次建索引会自动下载 bge 模型，请耐心等待")
    st.caption("文档存 `docs/`，索引存 `.rag_index/`，刷新页面不会丢")


# ================= 主区：数据看板 =================
doc_count = len(docs)
chunk_count = len(st.session_state["chunks"])
round_count = sum(1 for m in st.session_state["messages"] if m["role"] == "user")
render_dashboard(doc_count, chunk_count, round_count, top_k)


# ================= 主区：问答 =================
if st.session_state["index"] is not None:
    st.markdown("---")
    if st.session_state["messages"]:
        if st.button("🗑️ 清空对话"):
            st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔎 查看依据的原文片段"):
                    render_sources(msg["sources"])

    if prompt := st.chat_input("请输入问题（基于知识库文档）："):
        if not api_key:
            st.error("请先在左侧填 DeepSeek API Key")
        else:
            results = retrieve(st.session_state["index"], st.session_state["chunks"],
                               prompt, top_k)
            history = [(m["role"], m["content"]) for m in st.session_state["messages"]]

            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    generate_stream(prompt, results, api_key=api_key, history=history))
                st.caption(f"🎯 本轮检索命中 {len(results)} 个相关片段")
                with st.expander("🔎 查看依据的原文片段"):
                    render_sources(results)

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "sources": results})
else:
    st.info("👈 先在左侧上传文档、点「保存文档」再「重建索引」，然后就可以提问了。")
