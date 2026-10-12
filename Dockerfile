FROM python:3.11-slim

WORKDIR /app

# 先复制依赖清单，利用 Docker 层缓存：依赖没变就不重新下载
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 再复制源码（只带 API 运行时，不含网页版/命令行/测试）
COPY rag ./rag
COPY api.py .

# bge 模型缓存目录（用卷挂载，避免每次容器重建都重新下载模型）
ENV HF_HOME=/app/.hf_cache \
    HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
