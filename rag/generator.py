# -*- coding: utf-8 -*-
"""生成：把检索到的资料（+ 多轮对话历史）拼进 prompt，交给 DeepSeek 回答。

build_prompt 是纯逻辑，可脱离 openai 单独单测。
api_key 支持调用方覆盖（网页版用户手动填的 key 可以传进来，不必写进 .env）。
"""
import logging

from .config import API_KEY, BASE_URL, MODEL, TIMEOUT

logger = logging.getLogger("rag.generator")


def build_prompt(query, results, history=None):
    """纯逻辑：拼出防幻觉 prompt。

    results 是 [(相似度, {"file","text"}), ...]；
    history 是 [(role, content), ...]，role 为 user/assistant，多轮对话时传。
    """
    if results:
        context = "\n\n".join(
            f"[资料{idx + 1}] {r['text']}" for idx, (_, r) in enumerate(results)
        )
    else:
        context = "（没有检索到相关资料）"

    history_block = ""
    if history:
        lines = "\n".join(
            f"{'用户' if role == 'user' else '助手'}：{content}" for role, content in history
        )
        history_block = f"\n之前的对话：\n{lines}\n"

    return f"""请只根据下面提供的资料回答问题，资料里没有的内容就明确说不知道，不要编造。
{history_block}
资料：
{context}

问题：{query}

回答："""


def _resolve_key(api_key=None):
    key = api_key or API_KEY
    if not key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY（环境变量 / .env / key.txt 都未找到）"
        )
    return key


def _get_client(api_key=None):
    from openai import OpenAI
    return OpenAI(api_key=_resolve_key(api_key), base_url=BASE_URL, timeout=TIMEOUT)


def generate(query, results, temperature=0.3, api_key=None, history=None):
    """非流式生成，返回完整回答文本。"""
    prompt = build_prompt(query, results, history=history)
    resp = _get_client(api_key).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=False,
    )
    return resp.choices[0].message.content or ""


def generate_stream(query, results, temperature=0.3, api_key=None, history=None):
    """流式生成，逐段 yield 文本（供网页 / SSE 用）。"""
    prompt = build_prompt(query, results, history=history)
    stream = _get_client(api_key).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
