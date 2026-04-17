"""
RAGBench — LLM Retrieval Evaluation Framework
"""
from __future__ import annotations

import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.db.database import init_db
from app.core.ingest import ingest_folder
from app.core.hybrid import HybridRetriever

from app.routes import home, benchmark, dashboard, analysis, runs, compare, regression, upload

app = FastAPI(title="RAGBench", version="0.2.0")

# ── Global model cache — loads ONCE at startup ──
_cache: dict = {}

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Register all routes ──
app.include_router(home.router)
app.include_router(benchmark.router)
app.include_router(dashboard.router)
app.include_router(analysis.router)
app.include_router(runs.router)
app.include_router(compare.router)
app.include_router(regression.router)
app.include_router(upload.router)


@app.on_event("startup")
def _startup():
    init_db()
    print("[RAGBench] 🚀 Pre-loading Dense model at startup...")
    t0 = time.time()
    chunks = ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    chunk_map = {c.chunk_id: c.text for c in chunks}
    _cache["hybrid"]    = HybridRetriever(chunk_map)
    _cache["chunks"]    = chunks
    _cache["chunk_map"] = chunk_map
    elapsed = round((time.time() - t0) * 1000)
    print(f"[RAGBench] ✅ Model ready in {elapsed}ms")