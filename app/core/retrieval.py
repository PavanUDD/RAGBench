from dataclasses import dataclass
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    # simple, fast tokenizer (we can improve later)
    return [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split() if t.strip()]


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float


class BM25Retriever:
    def __init__(self, docs: Dict[str, str]):
        self.doc_ids = list(docs.keys())
        corpus = [docs[doc_id] for doc_id in self.doc_ids]
        tokenized = [_tokenize(t) for t in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs

    def search(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        pairs: List[Tuple[str, float]] = list(zip(self.doc_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:k]
        return [RetrievedDoc(doc_id=doc_id, score=float(score)) for doc_id, score in top]
