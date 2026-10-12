# -*- coding: utf-8 -*-
"""RAG 后端（FastAPI）：REST + SSE 流式 + 多轮会话记忆。

接口一览：
    GET    /health                 健康检查 + 当前是否已建索引
    POST   /build                  上传文档（multipart，txt/pdf/docx）建索引，并存盘
    POST   /load                   从 docs 文件夹批量建索引
    POST   /query                  一次性返回 {answer, sources}；带 session_id 则有记忆
    POST   /query/stream           SSE 流式返回答案；带 session_id 则有记忆
    DELETE /session/{session_id}   清空某个会话的历史

启动（在 learn_ai 目录下）：
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload
启动后浏览器打开 http://127.0.0.1:8000/docs 查看自动生成的接口文档。
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import (API_KEY, INDEX_DIR, TOP_K, add_message, build_index,
                 clear_session, generate, generate_stream, get_history,
                 load_index, retrieve, save_index, setup_logging)
from rag.loader import read_file_bytes

setup_logging()

# 进程内状态：当前生效的索引和文本块（POST /build 后更新）
_STATE = {"index": None, "chunks": []}


@asynccontextmanager
async def lifespan(_app):
    # 启动时尝试从磁盘读回上次存的索引，免重建
    try:
        index, chunks = load_index(INDEX_DIR)
        if index is not None:
            _STATE["index"] = index
            _STATE["chunks"] = chunks
    except Exception:
        pass
    yield


app = FastAPI(title="RAG 知识问答 API", version="0.2.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., description="问题")
    top_k: int = Field(TOP_K, ge=1, le=20, description="检索片段数")
    session_id: Optional[str] = Field(None, description="会话 ID，传了就启用多轮记忆")


class Source(BaseModel):
    score: float
    file: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


def _build_from_bytes(files: List[UploadFile]):
    docs = []
    for f in files:
        text = read_file_bytes(f.filename or "", f.file.read()).strip()
        if text:
            docs.append((f.filename, text))
    if not docs:
        raise ValueError("没有可入库的文字内容（请上传文字版 txt/pdf/docx）")
    return build_index(docs)


def _check_ready():
    if _STATE["index"] is None:
        raise HTTPException(status_code=400, detail="索引为空，请先 POST /build 或 /load 建索引")
    if not API_KEY:
        raise HTTPException(status_code=503, detail="未配置 DEEPSEEK_API_KEY（环境变量 / .env）")


@app.get("/health")
def health():
    return {"status": "ok", "indexed": len(_STATE["chunks"])}


@app.post("/build")
def build(files: List[UploadFile] = File(...)):
    try:
        index, chunks = _build_from_bytes(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    _STATE["index"] = index
    _STATE["chunks"] = chunks
    persisted = False
    try:
        save_index(index, chunks, INDEX_DIR)
        persisted = True
    except Exception:
        pass
    return {"ok": True, "chunks": len(chunks), "persisted": persisted}


@app.post("/load")
def load_from_folder(folder: Optional[str] = None):
    from rag import build_from_folder
    try:
        index, chunks = build_from_folder(folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _STATE["index"] = index
    _STATE["chunks"] = chunks
    return {"ok": True, "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    _check_ready()

    history = get_history(req.session_id) if req.session_id else []
    results = retrieve(_STATE["index"], _STATE["chunks"], req.question, req.top_k)
    answer = generate(req.question, results, history=history)

    if req.session_id:
        add_message(req.session_id, "user", req.question)
        add_message(req.session_id, "assistant", answer)

    sources = [Source(score=s, file=c["file"], text=c["text"]) for s, c in results]
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    _check_ready()

    history = get_history(req.session_id) if req.session_id else []
    results = retrieve(_STATE["index"], _STATE["chunks"], req.question, req.top_k)

    def gen():
        full = []
        for piece in generate_stream(req.question, results, history=history):
            full.append(piece)
            yield f"data: {piece}\n\n"
        if req.session_id:
            add_message(req.session_id, "user", req.question)
            add_message(req.session_id, "assistant", "".join(full))
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"ok": True, "session_id": session_id}
