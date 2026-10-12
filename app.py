# -*- coding: utf-8 -*-
"""RAG 网页版（Streamlit）——核心逻辑在 rag 包，这里只负责界面。

多轮记忆：同一浏览器会话里，模型会记住之前的问答（st.session_state 存对话，
生成时把历史拼进 prompt）。

运行（在 learn_ai 目录下）：
    streamlit run app.py
"""
import streamlit as st

from rag import API_KEY, EMBED_MODEL_NAME, build_index, retrieve, generate_stream
from rag.loader import read_file_bytes

# ================= 页面 =================
st.set_page_config(page_title="RAG 知识问答", page_icon="📚", layout="wide")
st.title("📚 RAG 知识问答系统")
st.caption("上传文档 → 建立语义索引 → 提问，答案会标注它依据的原文片段")

# ================= API Key（优先 st.secrets / .env，其次手动输入） =================
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
    st.caption(f"向量模型：`{EMBED_MODEL_NAME}`")
    st.caption("第一次运行会自动下载 bge 模型，耐心等")

# ================= 上传文档 → 建索引 =================
uploaded = st.file_uploader("上传文档（txt / pdf / docx，可多选）",
                            type=["txt", "pdf", "docx"],
                            accept_multiple_files=True)

if st.button("📥 建立索引", type="primary", disabled=not uploaded):
    with st.spinner("正在读取文档、建立语义索引……"):
        docs = []
        for f in uploaded:
            try:
                text = read_file_bytes(f.name, f.getvalue()).strip()
            except Exception as e:
                st.warning(f"{f.name} 读取失败：{e}")
                continue
            if text:
                docs.append((f.name, text))
            else:
                st.warning(f"{f.name} 没提取到文字（可能是扫描版 PDF），已跳过")
        if not docs:
            st.error("没有可用的文字，请换文字版文档（能选中复制文字的才算）")
        else:
            try:
                index, chunks = build_index(docs)
            except ImportError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"建索引失败：{e}")
            else:
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["messages"] = []
                st.success(f"✅ 已入库 {len(chunks)} 个文本片段，现在可以提问了")

# ================= 提问区（对话历史 + 流式输出 + 多轮记忆） =================
if "index" in st.session_state:
    st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []   # {"role", "content", "sources"}

    if st.session_state["messages"]:
        if st.button("🗑️ 清空对话"):
            st.session_state["messages"] = []
            st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔎 依据的原文片段"):
                    for idx, (score, c) in enumerate(msg["sources"]):
                        st.caption(f"片段 {idx + 1} · 相似度 {score:.3f} · 来自《{c['file']}》")
                        st.write(c["text"])

    if prompt := st.chat_input("请输入问题（基于已入库文档）："):
        if not api_key:
            st.error("请先在左侧填 DeepSeek API Key")
        else:
            results = retrieve(st.session_state["index"], st.session_state["chunks"],
                               prompt, top_k)
            # 把之前的对话历史传给模型，实现多轮记忆（能接着追问）
            history = [(m["role"], m["content"]) for m in st.session_state["messages"]]

            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    generate_stream(prompt, results, api_key=api_key, history=history))
                with st.expander("🔎 依据的原文片段"):
                    for idx, (score, c) in enumerate(results):
                        st.caption(f"片段 {idx + 1} · 相似度 {score:.3f} · 来自《{c['file']}》")
                        st.write(c["text"])

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "sources": results})
