from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float


class DenseRetriever:
    """
    FAISS-backed dense retriever using sentence-transformers embeddings.
    Matches resume: 'FAISS vector indexing + dense semantic embeddings'
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # fast, good quality, 384-dim

    def __init__(self, docs: Dict[str, str]):
        self.doc_ids = list(docs.keys())
        self.texts = [docs[d] for d in self.doc_ids]

        print(f"[DenseRetriever] Loading model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)

        print(f"[DenseRetriever] Encoding {len(self.texts)} chunks...")
        embeddings = self.model.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → cosine via inner product
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine (normalized)
        self.index.add(embeddings)
        print(f"[DenseRetriever] Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = 10) -> List[RetrievedDoc]:
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedDoc(doc_id=self.doc_ids[idx], score=float(score)))
        return results