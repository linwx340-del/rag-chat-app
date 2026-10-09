# RAG 知识问答系统

一个基于 **RAG（检索增强生成）** 的文档问答系统：上传 txt / pdf / docx 文档，自动建立语义索引，提问后由大模型基于检索到的原文片段生成答案，并标注答案依据的原文出处，有效减少「幻觉」。

## 功能特性

- 支持 txt、pdf、docx 三种文档格式
- 语义向量检索（bge 中文向量模型 + FAISS），按「意思」匹配，不靠关键词
- 答案防幻觉：只基于检索到的资料回答，资料里没有就明确说不知道
- 答案下方展示「依据的原文片段」及相似度，来源可追溯

## 技术栈

- 界面：Streamlit
- 向量化：sentence-transformers（`BAAI/bge-small-zh-v1.5`）
- 向量检索：FAISS（归一化向量内积 = 余弦相似度）
- 生成：DeepSeek（`deepseek-chat`）
- 文档解析：pdfplumber（PDF）、python-docx（Word）

## 本地运行

1. 安装依赖：

```bash
pip install streamlit sentence-transformers faiss-cpu openai pdfplumber python-docx
```

2. 启动：

```bash
streamlit run app.py
```

3. 浏览器打开（默认 http://localhost:8501），在左侧填 DeepSeek API Key → 上传文档 → 点「建立索引」→ 提问。

## 部署到 Streamlit Community Cloud

1. 把本仓库推送到 GitHub（公开仓库）。
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登录，点 **New app** 选择本仓库和 `app.py`。
3. 在 **Settings → Secrets** 里添加（**不要写进代码**）：

```toml
DEEPSEEK_API_KEY = "sk-你的key"
```

4. 点 **Deploy**，得到公开网址，发给别人即可使用。

## 项目结构

```
learn_ai/
├── app.py                  # 网页版（Streamlit）
├── main.py                 # 命令行版
├── ocr_pdf_to_word.py      # 扫描版 PDF 的 OCR 转 Word 脚本
├── .streamlit/
│   └── config.toml         # 主题配置（浅色）
└── README.md
```

## 效果截图

（运行后截图，替换下面的占位图）

![界面截图](screenshot.png)
