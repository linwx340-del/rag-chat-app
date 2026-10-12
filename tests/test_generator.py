# -*- coding: utf-8 -*-
"""generator 纯逻辑单测（build_prompt 不依赖 openai）。"""
from rag.generator import build_prompt


def _fake_results():
    return [
        (0.9, {"file": "a.txt", "text": "RAG 能减少幻觉。"}),
        (0.8, {"file": "b.txt", "text": "RAG 用向量检索。"}),
    ]


def test_prompt_contains_sources_and_query():
    prompt = build_prompt("RAG 有什么好处？", _fake_results())
    assert "[资料1]" in prompt
    assert "[资料2]" in prompt
    assert "RAG 能减少幻觉" in prompt
    assert "RAG 有什么好处？" in prompt


def test_prompt_has_anti_hallucination_instruction():
    prompt = build_prompt("RAG 有什么好处？", _fake_results())
    assert "不要编造" in prompt


def test_prompt_empty_results_does_not_crash():
    prompt = build_prompt("测试", [])
    assert "没有检索到相关资料" in prompt
    assert "测试" in prompt


def test_prompt_with_history():
    history = [("user", "什么是 RAG？"), ("assistant", "RAG 是检索增强生成。")]
    prompt = build_prompt("它有什么好处？", _fake_results(), history=history)
    assert "之前的对话" in prompt
    assert "什么是 RAG？" in prompt
    assert "它有什么好处？" in prompt


def test_prompt_without_history_has_no_history_block():
    prompt = build_prompt("问题", _fake_results())
    assert "之前的对话" not in prompt
