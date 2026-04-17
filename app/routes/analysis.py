"""
app/routes/analysis.py
"""
from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.retrieval import BM25Retriever
from app.core.benchmarks import build_benchmark_from_docs
from app.core.ingest import ingest_folder

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

DOC_LABELS = {
    "api_errors_legacy": "LEGACY/OUTDATED",
    "legacy_logging_guidelines": "LEGACY/OUTDATED",
    "logging_minimal_legacy": "LEGACY/OUTDATED",
    "incident_calendar_policy": "POLICY (NOT INCIDENT STEPS)",
    "security_logging_exceptions": "EXCEPTION POLICY",
}

def _get_cache():
    from app.main import _cache
    return _cache

@router.get("/analysis/{run_id}", response_class=HTMLResponse)
def analysis(run_id: str, request: Request, q: int = 0):
    cache = _get_cache()
    chunks    = cache.get("chunks")    or ingest_folder("data/docs", 120, 25)
    chunk_map = cache.get("chunk_map") or {c.chunk_id: c.text for c in chunks}
    bench = build_benchmark_from_docs(chunks)
    if not bench:
        return HTMLResponse("<h3>No benchmark queries found.</h3>")

    selected_idx = max(0, min(q, len(bench)-1))
    selected = bench[selected_idx]
    results = BM25Retriever(chunk_map).search(selected.query, k=10)
    relevant = set(selected.relevant_chunk_ids)
    rows, hit, hit_rank = [], False, None

    for i, r in enumerate(results, 1):
        is_rel = r.doc_id in relevant
        if is_rel and not hit:
            hit, hit_rank = True, i
        txt = chunk_map.get(r.doc_id, "")
        rows.append({"rank":i,"chunk_id":r.doc_id,"score":r.score,
                     "preview":txt[:220]+("..." if len(txt)>220 else ""),
                     "is_relevant":is_rel,"label":DOC_LABELS.get(r.doc_id.split("::")[0],"CURRENT")})

    why = ""
    if not hit and relevant:
        rel_text = chunk_map.get(list(relevant)[0], "")
        q_terms = set(t.lower().strip(".,!?;:()") for t in selected.query.split())
        d_terms = set(t.lower().strip(".,!?;:()") for t in rel_text.split())
        overlap = sorted(q_terms & d_terms)
        why = (f"Some overlap exists ({', '.join(overlap)}), but other chunks scored higher."
               if overlap else
               "Low lexical overlap — query terms don't appear in the relevant chunk. Try hybrid retrieval.")

    return templates.TemplateResponse("analysis.html", {
        "request":request,"run_id":run_id,"queries":bench,
        "selected_idx":selected_idx,"selected":selected,"k_rank":10,
        "relevant_count":len(relevant),
        "relevant_ids":", ".join(sorted(relevant)) if relevant else "(none)",
        "retrieved":rows,"hit":hit,"hit_rank":hit_rank,"why":why,
    })
