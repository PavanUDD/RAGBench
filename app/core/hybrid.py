from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.core.retrieval import BM25Retriever
from app.core.dense import DenseRetriever


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float


class HybridRetriever:
    """
    Hybrid retriever: BM25 (lexical) + Dense FAISS (semantic).
    Matches resume: 'hybrid ranking engine combining BM25 + dense embeddings'
    Fusion method: Reciprocal Rank Fusion (RRF) — standard industry approach.
    """

    def __init__(self, docs: Dict[str, str], rrf_k: int = 60):
        self.rrf_k = rrf_k
        print("[HybridRetriever] Building BM25 index...")
        self.bm25 = BM25Retriever(docs)
        print("[HybridRetriever] Building Dense FAISS index...")
        self.dense = DenseRetriever(docs)

    def search(self, query: str, k: int = 10) -> List[RetrievedDoc]:
        # Get candidates from both retrievers (fetch 2x for better fusion coverage)
        fetch_k = min(k * 2, 20)
        bm25_results  = self.bm25.search(query, k=fetch_k)
        dense_results = self.dense.search(query, k=fetch_k)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}

        for rank, r in enumerate(bm25_results, start=1):
            rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, r in enumerate(dense_results, start=1):
            rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [RetrievedDoc(doc_id=doc_id, score=score) for doc_id, score in ranked]