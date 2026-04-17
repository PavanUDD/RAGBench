from __future__ import annotations
import time
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.retrieval import BM25Retriever
from app.core.tfidf import TfidfRetriever
from app.core.benchmarks import build_benchmark_from_docs
from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from app.core.ingest import ingest_folder
from app.db.database import get_conn

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _get_cache():
    from app.main import _cache
    return _cache


def _total_runs():
    try:
        conn = get_conn()
        n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def _eval(retriever, bench, k_recall=5, k_rank=10):
    recalls, mrrs, ndcgs, lats = [], [], [], []
    for bq in bench:
        t0 = time.time()
        results = retriever.search(bq.query, k=k_rank)
        lats.append((time.time() - t0) * 1000)
        ids = [r.doc_id for r in results]
        rel = set(bq.relevant_chunk_ids)
        recalls.append(recall_at_k(rel, ids, k=k_recall))
        mrrs.append(mrr_at_k(rel, ids, k=k_rank))
        ndcgs.append(ndcg_at_k(rel, ids, k=k_rank))
    avg = lambda xs: round(sum(xs) / max(len(xs), 1), 4)
    return {"recall5": avg(recalls), "mrr10": avg(mrrs), "ndcg10": avg(ndcgs),
            "latency_ms": round(sum(lats) / len(lats), 1)}


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    cache = _get_cache()
    chunks    = cache.get("chunks")    or ingest_folder("data/docs", 120, 25)
    chunk_map = cache.get("chunk_map") or {c.chunk_id: c.text for c in chunks}
    hybrid    = cache.get("hybrid")

    bench = build_benchmark_from_docs(chunks)
    bm25_d   = {"name": "BM25",   **_eval(BM25Retriever(chunk_map),  bench)}
    tfidf_d  = {"name": "TFIDF",  **_eval(TfidfRetriever(chunk_map), bench)}
    hybrid_d = {"name": "HYBRID", **_eval(hybrid, bench)}

    winner = max([bm25_d, tfidf_d, hybrid_d], key=lambda x: x["mrr10"])["name"]
    bm25_base = bm25_d["mrr10"]

    CFG = {
        "HYBRID": ("#a78bfa","bar-purple","tag-winner" if winner=="HYBRID" else "tag-strong", "🏆 Winner" if winner=="HYBRID" else "Hybrid","BM25 + Dense Embeddings (RRF)"),
        "TFIDF":  ("#60a5fa","bar-blue",  "tag-winner" if winner=="TFIDF"  else "tag-strong", "🏆 Winner" if winner=="TFIDF"  else "Strong", "TF-IDF Cosine Similarity"),
        "BM25":   ("#34d399","bar-green", "tag-winner" if winner=="BM25"   else "tag-baseline","🏆 Winner" if winner=="BM25"   else "Baseline","BM25 Okapi Lexical"),
    }

    def ctx(d):
        color,bar,tag_cls,tag_lbl,method = CFG[d["name"]]
        delta = round(d["mrr10"] - bm25_base, 4)
        return {**d, "is_winner": d["name"]==winner, "color":color, "bar_class":bar,
                "tag_class":tag_cls, "tag_label":tag_lbl, "method":method,
                "score_class":"score-good" if d["name"]==winner else ("score-ok" if d["name"]=="TFIDF" else "score-base"),
                "recall5_pct": round(d["recall5"]*100,1), "mrr10_pct": round(d["mrr10"]*100,1),
                "ndcg10_pct": round(d["ndcg10"]*100,1), "delta_mrr": delta, "delta_mrr_str": f"{abs(delta):.4f}"}

    w = max([bm25_d, tfidf_d, hybrid_d], key=lambda x: x["mrr10"])
    gap = round(w["mrr10"] - bm25_base, 3)
    verdicts = {
        "HYBRID": ("Combining BM25 lexical matching with dense semantic embeddings via RRF captures both exact keyword hits and conceptual similarity.",
                   "Deploy HYBRID for production. Use TFIDF as a lightweight fallback if latency is critical."),
        "TFIDF":  ("TF-IDF outperforms on this keyword-dense corpus — semantic similarity adds less signal when vocabulary is highly consistent.",
                   "Deploy TFIDF for this corpus. Re-evaluate with a more semantically diverse document set."),
        "BM25":   ("BM25 leads — benchmark queries use terminology that directly matches document content with high lexical precision.",
                   "BM25 is sufficient here. Monitor as corpus grows — semantic retrievers gain edge with diversity."),
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "winner": winner, "winner_mrr": w["mrr10"],
        "verdict_summary": f"{winner} achieves MRR@10 of {w['mrr10']} — outperforming BM25 baseline by {gap} across {len(bench)} queries and {len(chunk_map)} chunks.",
        "verdict_why": verdicts[winner][0], "verdict_rec": verdicts[winner][1],
        "bm25_gap": f"{gap:.3f}",
        "retrievers": [ctx(hybrid_d), ctx(tfidf_d), ctx(bm25_d)],
        "total_runs": _total_runs(), "doc_count": len(chunk_map), "query_count": len(bench),
    })
