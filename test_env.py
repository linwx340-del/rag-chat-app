"""
RAG 精简版 —— 从零手写，不依赖任何 RAG 框架
====================================================

一个 RAG 系统就五个环节，这个脚本一个不落地全写出来：

    1) 加载文档    load      → 读入文本（内置示例 / 或你自己的 .txt 文件）
    2) 切块        chunk     → 把长文按固定字数切成小块，相邻块留重叠
    3) 向量化      embed     → 把每块文字变成一串数字（向量）
                               这里用 TF-IDF 代替神经网络，无需下载模型、零门槛
    4) 检索        retrieve  → 把你的问题也变成向量，用余弦相似度找最相近的几块
    5) 生成        generate  → 把"检索到的资料 + 你的问题"拼进 prompt，交给 DeepSeek 回答

依赖： numpy + openai
    pip install numpy openai

运行（交互式提问，用内置示例知识库）：
    python rag_demo.py

加载你自己的文件：
    python rag_demo.py 你的文件.txt

DeepSeek 的 API key 用环境变量传入（推荐，避免写死在代码里）：
    Windows PowerShell:  $env:DEEPSEEK_API_KEY="sk-xxxx"; python rag_demo.py
    Windows CMD:         set DEEPSEEK_API_KEY=sk-xxxx && python rag_demo.py
"""

import os
import re
import sys

import numpy as np

# ==================== 配置区 ====================
# 推荐用环境变量 DEEPSEEK_API_KEY 传 key；如果只是想快速跑，也可以直接填在下面引号里。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")   # 例： "sk-xxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"          # DeepSeek 是 OpenAI 兼容接口
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 120      # 每块最多多少字
CHUNK_OVERLAP = 20    # 相邻两块重叠多少字（避免一句话被拦腰切断）
TOP_K = 3             # 检索时取相似度最高的几块
# ===============================================


# ---------------------------------------------------------------
# 第 1 步：加载文档。内置了一份关于 RAG 的示例知识库，你也可以换成自己的文件。
# ---------------------------------------------------------------
CORPUS = [
    "检索增强生成（RAG）是一种把信息检索和大语言模型结合起来的技术。核心思路是：在回答用户问题之前，先从外部知识库里检索出相关的文档片段，再把这些片段作为上下文提供给大模型，让模型基于资料来生成答案，而不是凭空回答。",
    "RAG 最大的好处是能缓解大模型的'幻觉'问题，并且让模型回答那些训练数据里没有的、时效性强的内容。因为答案的依据来自外部资料，而不是模型自己脑子里的记忆。",
    "一个典型的 RAG 系统包含五个步骤：加载文档、把文档切成小块、把每个块向量化、根据用户问题检索最相关的块、最后把问题和检索到的块一起交给大模型生成答案。",
    "向量化（embedding）是把一段文本变成一组数字向量的过程。语义相近的文本，它们对应的向量在空间里的距离也更近。常见的向量化方法有 TF-IDF、Word2Vec，以及现在常用的预训练模型如 BERT、bge 等。",
    "向量数据库（比如 FAISS、Milvus、Chroma）专门用来存储和检索大量向量，支持快速的近似最近邻搜索。在 RAG 里，向量数据库负责根据查询向量找出最相似的文档块。",
    "切块（chunking）是 RAG 里很关键的一步。切得太大，检索出来的块会带进噪声；切得太小，又会丢失上下文。通常会在相邻块之间保留一定重叠，避免一个完整的意思被切断。",
    "DeepSeek 是一家中国 AI 公司，它的 DeepSeek 系列大模型以高性能和低成本著称。DeepSeek 提供了 OpenAI 兼容的 API 接口，开发者可以直接用 openai 的 SDK 来调用它。",
    "提示词工程（prompt engineering）是指通过精心设计输入给模型的提示词，来引导模型产生更准确、更符合预期的结果。在 RAG 里，通常会把检索到的资料放进提示词，并明确要求模型只依据这些资料回答。",
]


def load_documents(argv):
    """返回要检索的文档列表。优先加载命令行给的 .txt 文件，否则用内置示例。"""
    if len(argv) >= 2 and os.path.isfile(argv[1]):
        with open(argv[1], encoding="utf-8") as f:
            print(f"已加载文件：{argv[1]}")
            return [f.read()]
    print("使用内置示例知识库（内容是关于 RAG 的介绍）。")
    print('提示：想检索自己的文件，运行  python rag_demo.py 你的文件.txt\n')
    return list(CORPUS)


# ---------------------------------------------------------------
# 第 2 步：切块。把长文本按 CHUNK_SIZE 切成块，相邻块重叠 CHUNK_OVERLAP。
# ---------------------------------------------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------
# 第 3 步：向量化。这里用 TF-IDF 把文本变成向量。
#   - TF（词频）：一个词在这段文字里出现多少次
#   - IDF（逆文档频率）：一个词越"稀罕"、越有区分度，权重越高
#   - 一句话的向量 = 每个词 TF 值 × IDF 值 组成的一串数字
#
# 中文没有空格，所以这里把汉字拆成"单字 + 相邻二元组"当作词，英文数字拆成整词。
# 同时过滤掉"的/了/是/它/有"这类没信息量的停用词——否则它们在文档里很少出现、
# IDF 反而很高，会严重干扰排序（这是小语料下 TF-IDF 的典型坑）。
#
# 注意：TF-IDF 只能匹配"字面相同"的词，理解不了"优点"和"好处"是同义词。
# 这是它和真正语义向量（embedding）的差距。下一步的升级方向：
#   用 sentence-transformers + bge-small-zh 把 TF-IDF 换成神经网络向量，
#   就能解决同义词问题。学会这个脚本的流程后，替换"向量化+检索"这一层即可。
# ---------------------------------------------------------------
STOPWORDS = set(
    "的一了是我不他她它你您们在有这那和与及或而把被让对就都也很个之其该"
    "什么怎么为什么一个一种这种那种如何哪些吧吗呢啊呀哦但若如从而因为所以于是然后却只又再"
)


def tokenize(text):
    text = text.lower()
    tokens = []
    for m in re.finditer(r"[a-z0-9]+", text):       # 英文 / 数字 → 整词
        w = m.group()
        if w not in ("the", "a", "an", "of", "to", "is", "are", "in", "on", "for", "and", "or"):
            tokens.append(w)
    cjk = [c for c in re.findall(r"[一-鿿]", text) if c not in STOPWORDS]  # 汉字 → 单字 + 二元组
    tokens += cjk
    tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens


class TfidfRetriever:
    """一个极简的 TF-IDF 检索器，学会它你就理解了向量检索的本质。"""

    def __init__(self):
        self.vocab = []            # 词表
        self.idf = None            # 每个词的 IDF 权重
        self.doc_vectors = None    # 每个文档块归一化后的向量

    def fit(self, doc_tokens_list):
        """根据一批文档块，算出词表、IDF，以及每块的向量。"""
        # 1) 统计词频，建词表
        counter = {}
        for toks in doc_tokens_list:
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
        self.vocab = list(counter.keys())
        v2i = {w: i for i, w in enumerate(self.vocab)}

        n_docs = len(doc_tokens_list)

        # 2) 文档频率 df：一个词出现在多少个块里
        df = np.zeros(len(self.vocab))
        for toks in doc_tokens_list:
            for t in set(toks):
                if t in v2i:
                    df[v2i[t]] += 1

        # 3) IDF：出现越少的词权重越高（+1 平滑，避免除零）
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        # 4) 每块的 TF-IDF 向量，再做 L2 归一化（归一化后，点积就等于余弦相似度）
        self.doc_vectors = np.zeros((n_docs, len(self.vocab)))
        for i, toks in enumerate(doc_tokens_list):
            for t in toks:
                if t in v2i:
                    self.doc_vectors[i, v2i[t]] += 1     # 这就是 TF
        self.doc_vectors *= self.idf                     # TF × IDF
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_vectors /= norms

    # ---------------------------------------------------------------
    # 第 4 步：检索。把问题也变成向量，和每块算余弦相似度，取最高的 TOP_K 块。
    # ---------------------------------------------------------------
    def search(self, query, k=TOP_K):
        v2i = {w: i for i, w in enumerate(self.vocab)}
        qv = np.zeros(len(self.vocab))
        for t in tokenize(query):
            if t in v2i:
                qv[v2i[t]] += 1
        qv = qv * self.idf
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm

        scores = self.doc_vectors @ qv          # 归一化后点积 = 余弦相似度
        top_idx = np.argsort(-scores)[:k]       # 从大到小排序，取前 k 个
        return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------
# 第 5 步：生成。把检索到的资料 + 问题拼进 prompt，调用 DeepSeek。
# ---------------------------------------------------------------
def generate(question, chunks):
    context = "\n\n".join(f"[资料{i + 1}]\n{text}" for i, (text, _) in enumerate(chunks))
    system = ("你是一个严谨的问答助手。请只根据用户提供的【资料】来回答问题；"
              "如果资料里没有相关信息，请直接说'资料中没有找到答案'，不要编造。")
    user = f"【资料】\n{context}\n\n【问题】\n{question}"

    # 没有 key 时进入"演示模式"：不调模型，只把要发的内容打印出来，方便理解流程。
    if not DEEPSEEK_API_KEY:
        print("\n" + "=" * 60)
        print("[演示模式] 未检测到 DEEPSEEK_API_KEY，已跳过模型调用。")
        print("下面就是正常情况下要发给 DeepSeek 的内容：\n")
        print("--- SYSTEM 提示词 ---")
        print(system)
        print("\n--- USER 提示词 ---")
        print(user)
        print("=" * 60)
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("\n缺少 openai 库，请先运行： pip install openai")
        return

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    # 切块
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d))
    print(f"切块完成，共 {len(all_chunks)} 块\n")

    # 建检索器
    retriever = TfidfRetriever()
    retriever.fit([tokenize(c) for c in all_chunks])

    # 交互式问答
    while True:
        q = input("请输入问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        if not q:
            continue

        hits = retriever.search(q, k=TOP_K)
        print(f"\n检索到 {len(hits)} 个相关块：")
        for i, (idx, score) in enumerate(hits):
            preview = all_chunks[idx][:40].replace("\n", " ")
            print(f"  [{i + 1}] 相似度={score:.3f}  |  {preview}...")

        generate(q, [(all_chunks[idx], score) for idx, score in hits])


if __name__ == "__main__":
    main()
    print("\n--- DeepSeek 回答 ---\n")
    print(resp.choices[0].message.content)


def main():
    docs = load_documents(sys.argv)

