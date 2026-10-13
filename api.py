# -*- coding: utf-8 -*-
"""RAG 后端（FastAPI）：REST + SSE 流式 + 多租户 + 多轮会话记忆。

多租户（半产品化）：
    - 未配置 tenants.json 时退回单租户模式（无鉴权，行为与旧版一致）。
    - 配置 tenants.json 后启用多租户：每个请求带 X-API-Key 头，
      admin 角色可建索引 / 删会话，user 角色只能问答；索引与会话按租户隔离。

接口一览：
    GET    /health                 健康检查 + 各租户是否已建索引
    POST   /build                  上传文档（multipart）建索引并存盘（admin）
    POST   /load                   从文件夹批量建索引（admin）
    POST   /query                  一次性返回 {answer, sources}；带 session_id 有记忆
    POST   /query/stream           SSE 流式返回答案；带 session_id 有记忆
    DELETE /session/{session_id}   清空当前租户某个会话的历史

启动（在 learn_ai 目录下）：
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload
启动后浏览器打开 http://127.0.0.1:8000/docs 查看接口文档。
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import (API_KEY, INDEX_DIR, TOP_K, add_message, build_index,
                 clear_session, generate, generate_stream, get_history,
                 load_index, retrieve, save_index, setup_logging)
from rag.loader import read_file_bytes
from rag.tenants import (ADMIN_ROLE, get_tenant, index_dir_for, load_tenants,
                         tenant_scope)

setup_logging()

# 进程内状态：{scope: {"index": ..., "chunks": [...]}}，按租户知识库隔离
_STATE = {"tenants": {}}


def _multi_tenant_enabled():
    return bool(load_tenants())


def _resolve(x_api_key):
    """鉴权：返回 (tenant_id, tenant)。未配置 tenants.json 时退回单租户无鉴权。"""
    tenants = load_tenants()
    if not tenants:
        return "", {"role": ADMIN_ROLE}   # 单租户模式
    if not x_api_key:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key 请求头")
    tenant_id, tenant = get_tenant(x_api_key, tenants)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return tenant_id, tenant


def _require_admin(tenant):
    if tenant.get("role") != ADMIN_ROLE:
        raise HTTPException(status_code=403, detail="只有 admin 角色可执行此操作")


def _scope_of(tenant_id, tenant):
    """索引存储 scope：多租户用 kb（可共享），单租户用 ""（老路径）。"""
    if not tenant_id:
        return ""
    return tenant_scope(tenant_id, tenant)


def _state_of(scope):
    st = _STATE["tenants"].get(scope)
    if st is None:
        st = _STATE["tenants"][scope] = {"index": None, "chunks": []}
    return st


def _check_ready(st):
    if st["index"] is None:
        raise HTTPException(status_code=400, detail="该租户尚未建索引，请先 POST /build 或 /load")
    if not API_KEY:
        raise HTTPException(status_code=503, detail="未配置 DEEPSEEK_API_KEY（环境变量 / .env）")


@asynccontextmanager
async def lifespan(_app):
    # 启动时从磁盘读回各租户的索引，免重建
    try:
        tenants = load_tenants()
        if tenants:
            for tid, t in tenants.items():
                scope = tenant_scope(tid, t)
                index, chunks = load_index(index_dir_for(scope))
                if index is not None:
                    _STATE["tenants"][scope] = {"index": index, "chunks": chunks}
        else:
            index, chunks = load_index(INDEX_DIR)
            if index is not None:
                _STATE["tenants"][""] = {"index": index, "chunks": chunks}
    except Exception:
        pass
    yield


app = FastAPI(title="RAG 知识问答 API（多租户）", version="0.3.0", lifespan=lifespan)


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


@app.get("/health")
def health():
    indexed = {s: len(st["chunks"]) for s, st in _STATE["tenants"].items() if st["index"] is not None}
    return {"status": "ok", "multi_tenant": _multi_tenant_enabled(), "indexed": indexed}


@app.post("/build")
def build(files: List[UploadFile] = File(...), x_api_key: Optional[str] = Header(None)):
    tenant_id, tenant = _resolve(x_api_key)
    _require_admin(tenant)
    try:
        index, chunks = _build_from_bytes(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    scope = _scope_of(tenant_id, tenant)
    st = _state_of(scope)
    st["index"], st["chunks"] = index, chunks
    persisted = False
    try:
        save_index(index, chunks, index_dir_for(scope))
        persisted = True
    except Exception:
        pass
    return {"ok": True, "tenant": tenant_id or "default", "scope": scope or "default",
            "chunks": len(chunks), "persisted": persisted}


@app.post("/load")
def load_from_folder(folder: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    from rag import build_from_folder
    tenant_id, tenant = _resolve(x_api_key)
    _require_admin(tenant)
    try:
        index, chunks = build_from_folder(folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    scope = _scope_of(tenant_id, tenant)
    st = _state_of(scope)
    st["index"], st["chunks"] = index, chunks
    save_index(index, chunks, index_dir_for(scope))
    return {"ok": True, "tenant": tenant_id or "default", "scope": scope or "default",
            "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    tenant_id, tenant = _resolve(x_api_key)
    scope = _scope_of(tenant_id, tenant)
    st = _state_of(scope)
    _check_ready(st)

    history = get_history(req.session_id, tenant_id=tenant_id) if req.session_id else []
    results = retrieve(st["index"], st["chunks"], req.question, req.top_k)
    answer = generate(req.question, results, history=history)

    if req.session_id:
        add_message(req.session_id, "user", req.question, tenant_id=tenant_id)
        add_message(req.session_id, "assistant", answer, tenant_id=tenant_id)

    sources = [Source(score=s, file=c["file"], text=c["text"]) for s, c in results]
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
def query_stream(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    tenant_id, tenant = _resolve(x_api_key)
    scope = _scope_of(tenant_id, tenant)
    st = _state_of(scope)
    _check_ready(st)

    history = get_history(req.session_id, tenant_id=tenant_id) if req.session_id else []
    results = retrieve(st["index"], st["chunks"], req.question, req.top_k)

    def gen():
        full = []
        for piece in generate_stream(req.question, results, history=history):
            full.append(piece)
            yield f"data: {piece}\n\n"
        if req.session_id:
            add_message(req.session_id, "user", req.question, tenant_id=tenant_id)
            add_message(req.session_id, "assistant", "".join(full), tenant_id=tenant_id)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")


@app.delete("/session/{session_id}")
def delete_session(session_id: str, x_api_key: Optional[str] = Header(None)):
    tenant_id, _ = _resolve(x_api_key)
    clear_session(session_id, tenant_id=tenant_id)
    return {"ok": True, "session_id": session_id, "tenant": tenant_id or "default"}
