"""模块 2b：向量化 + 本地向量库 + top-k 检索（LlamaIndex + Chroma + bge-small-zh）。

每种 chunk 大小单独建一个 Chroma collection（chunks_300 / chunks_800），用于对比实验。
入库文本前加 "【公司 第N页】" 前缀：三家公司的年报措辞高度相似（都有"营业收入""净利润"），
不加公司名时，问宁德时代的营收可能检索到比亚迪的表格。
检索时如果问题里出现公司名，再用 metadata 过滤只在该公司年报里找。

建索引：python -m src.index build
试检索：python -m src.index search "宁德时代2025年营业收入是多少"
"""
import sys

import chromadb
import torch
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.chunking import load_chunks
from src.config import (CHROMA_DIR, CHUNK_SIZES, COMPANIES, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K,
                        EMBED_MODEL, EMBED_QUERY_INSTRUCTION)

_embed_model = None
_indexes: dict[int, VectorStoreIndex] = {}


def get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
            query_instruction=EMBED_QUERY_INSTRUCTION,
            embed_batch_size=64,
        )
    return _embed_model


def _vector_store(chunk_size: int) -> ChromaVectorStore:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(f"chunks_{chunk_size}", metadata={"hnsw:space": "cosine"})
    return ChromaVectorStore(chroma_collection=collection)


def build(chunk_size: int) -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = f"chunks_{chunk_size}"
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)  # 重建，保证和最新的解析结果一致

    nodes = []
    for c in load_chunks(chunk_size):
        node = TextNode(
            id_=c["chunk_id"],
            text=f"【{c['company']} 第{c['page']}页】\n{c['text']}",
            metadata={"doc": c["doc"], "company": c["company"], "page": c["page"]},
        )
        # 元数据不参与向量化，也不重复塞给大模型（来源信息由我们自己拼）
        node.excluded_embed_metadata_keys = list(node.metadata)
        node.excluded_llm_metadata_keys = list(node.metadata)
        nodes.append(node)

    storage = StorageContext.from_defaults(vector_store=_vector_store(chunk_size))
    VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model(), show_progress=True)
    return len(nodes)


def get_index(chunk_size: int) -> VectorStoreIndex:
    if chunk_size not in _indexes:
        store = _vector_store(chunk_size)
        if store._collection.count() == 0:  # 空库时检索结果为空，模型只会答"未披露"，必须显式报错
            raise SystemExit(f"向量库 chunks_{chunk_size} 为空（位置：{CHROMA_DIR}），"
                             f"请先运行：python -m src.index build")
        _indexes[chunk_size] = VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model())
    return _indexes[chunk_size]


def detect_company(question: str) -> str | None:
    hits = [c for c in COMPANIES if c in question or (c == "工商银行" and "工行" in question)]
    return hits[0] if len(hits) == 1 else None


def retrieve(question: str, chunk_size: int = DEFAULT_CHUNK_SIZE, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """返回 [{"doc", "company", "page", "text", "score"}]，按相似度从高到低。"""
    company = detect_company(question)
    filters = MetadataFilters(filters=[ExactMatchFilter(key="company", value=company)]) if company else None
    retriever = get_index(chunk_size).as_retriever(similarity_top_k=top_k, filters=filters)
    return [
        {
            "doc": r.node.metadata["doc"],
            "company": r.node.metadata["company"],
            "page": r.node.metadata["page"],
            "text": r.node.get_content(),
            "score": round(r.score, 4),
        }
        for r in retriever.retrieve(question)
    ]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        print("Embedding 设备：", "cuda" if torch.cuda.is_available() else "cpu")
        for size in CHUNK_SIZES:
            print(f"chunk_size={size}: 入库 {build(size):,} 个 chunk")
    elif cmd == "search" and len(sys.argv) > 2:
        for r in retrieve(sys.argv[2]):
            print(f"--- {r['doc']} 第{r['page']}页  score={r['score']}")
            print(r["text"][:200])
    else:
        print(__doc__)
