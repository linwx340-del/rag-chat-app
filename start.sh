#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  RAG 知识问答系统 - 一键启动"
echo "============================================"

# 1) 检查 .env
if [ ! -f .env ]; then
  echo "未找到 .env，正在从模板创建..."
  cp .env.example .env
  echo "请先打开 .env，把 sk-xxx 换成你的真实 DeepSeek Key，然后重新运行本脚本。"
  exit 1
fi

# 2) 检查依赖
if ! python -c "import streamlit, sentence_transformers, faiss, openai, dotenv" >/dev/null 2>&1; then
  echo "正在安装依赖（阿里云镜像）..."
  python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
fi

# 3) 选择启动方式
echo
echo "请选择启动方式：[1] 网页版（Streamlit） [2] API 后端（FastAPI）"
read -r -p "输入 1 或 2 后回车：" choice
case "$choice" in
  1) streamlit run app.py ;;
  2) uvicorn api:app --reload ;;
  *) echo "无效输入，默认启动网页版"; streamlit run app.py ;;
esac
