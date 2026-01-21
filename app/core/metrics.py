import math
from typing import List, Set


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / float(len(relevant))


def mrr_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    topk = retrieved[:k]
    for i, doc_id in enumerate(topk, start=1):
        if doc_id in relevant:
            return 1.0 / float(i)
    return 0.0


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    # Binary relevance: rel=1 if doc in relevant else 0
    topk = retrieved[:k]

    def dcg(items: List[str]) -> float:
        s = 0.0
        for i, doc_id in enumerate(items, start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            s += rel / math.log2(i + 1)
        return s

    ideal = list(relevant)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(topk) / idcg
