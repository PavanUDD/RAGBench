from __future__ import annotations
import json
import re
import random
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import HTMLResponse

from app.db.database import get_conn
from app.core.run_id import new_run_id
from app.core.ingest import chunk_text, Chunk
from app.core.retrieval import BM25Retriever, RetrievedDoc
from app.core.tfidf import TfidfRetriever
from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from app.core.benchmarks import BenchmarkQuery
from app.core.extractor import extract_text

router = APIRouter()


@router.get("/upload", response_class=HTMLResponse)
def upload_page():
    return HTMLResponse(Path("app/templates/upload.html").read_text(encoding="utf-8"))


@router.post("/upload-benchmark", response_class=HTMLResponse)
async def upload_benchmark(
    file: UploadFile = File(...),
    chunk_size: int = Form(200),
    overlap: int = Form(30),
    top_k: int = Form(5),
    num_queries: int = Form(20),
):
    # ── Extract ──────────────────────────────────────────────
    content = await file.read()
    try:
        raw_text = extract_text(file.filename, content)
    except ValueError as e:
        return HTMLResponse(str(e), status_code=400)

    if len(raw_text.strip()) < 100:
        return HTMLResponse("Document too short or could not be parsed.", status_code=400)

    # ── Chunk ─────────────────────────────────────────────────
    doc_id = re.sub(r"[^a-zA-Z0-9_]", "_", Path(file.filename).stem)[:40]
    text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    chunks = [Chunk(chunk_id=f"{doc_id}::c{i:03d}", doc_id=doc_id, text=t) for i, t in enumerate(text_chunks)]

    if len(chunks) < 3:
        return HTMLResponse("Document too short — need at least 3 chunks. Try smaller chunk size.", status_code=400)

    chunk_map = {c.chunk_id: c.text for c in chunks}

    # ── Auto-generate benchmark queries ──────────────────────
    sentences = [p.strip() for p in raw_text.split("\n") if 60 < len(p.strip()) < 300 and not p.strip().startswith("#")]
    random.seed(42)
    selected = random.sample(sentences, min(num_queries, len(sentences)))

    bench = []
    for sent in selected:
        relevant = {c.chunk_id for c in chunks if sent.lower()[:40] in c.text.lower()}
        if relevant:
            bench.append(BenchmarkQuery(query=sent[:200], relevant_chunk_ids=relevant))

    if len(bench) < 5:
        bench = [BenchmarkQuery(query=" ".join(c.text.split()[:8]), relevant_chunk_ids={c.chunk_id})
                 for c in chunks[:num_queries] if len(c.text.split()) >= 4]

    if not bench:
        return HTMLResponse("Could not generate benchmark queries from this document.", status_code=400)

    # ── Build Dense + Hybrid ──────────────────────────────────
    from app.core.dense import DenseRetriever

    dense_r = DenseRetriever(chunk_map)
    bm25_r  = BM25Retriever(chunk_map)
    tfidf_r = TfidfRetriever(chunk_map)

    class _Hybrid:
        def search(self, query, k=10):
            b = bm25_r.search(query, k=k*2)
            d = dense_r.search(query, k=k*2)
            sc: dict = {}
            for rank, r in enumerate(b, 1): sc[r.doc_id] = sc.get(r.doc_id,0) + 1/(60+rank)
            for rank, r in enumerate(d, 1): sc[r.doc_id] = sc.get(r.doc_id,0) + 1/(60+rank)
            return [RetrievedDoc(doc_id=d, score=s) for d,s in sorted(sc.items(), key=lambda x:-x[1])[:k]]

    hybrid_r = _Hybrid()

    # ── Evaluate ──────────────────────────────────────────────
    k_rank = 10

    def _eval(ret):
        recalls, mrrs, ndcgs = [], [], []
        for bq in bench:
            ids = [r.doc_id for r in ret.search(bq.query, k=k_rank)]
            rel = set(bq.relevant_chunk_ids)
            recalls.append(recall_at_k(rel, ids, k=top_k))
            mrrs.append(mrr_at_k(rel, ids, k=k_rank))
            ndcgs.append(ndcg_at_k(rel, ids, k=k_rank))
        avg = lambda xs: round(sum(xs)/max(len(xs),1), 4)
        return avg(recalls), avg(mrrs), avg(ndcgs)

    bm25_rec,   bm25_mrr,   bm25_ndcg   = _eval(bm25_r)
    tfidf_rec,  tfidf_mrr,  tfidf_ndcg  = _eval(tfidf_r)
    hybrid_rec, hybrid_mrr, hybrid_ndcg = _eval(hybrid_r)

    scores = {"HYBRID": hybrid_mrr, "TFIDF": tfidf_mrr, "BM25": bm25_mrr}
    winner = max(scores, key=scores.get)

    # ── Save to DB ────────────────────────────────────────────
    sig = {"doc": doc_id, "chunks": len(chunks), "queries": len(bench)}
    cfg_base = {"dataset": f"upload:{file.filename}", "chunking": {"chunk_size":chunk_size,"overlap":overlap},
                "k_recall":top_k, "k_rank":k_rank, "source":"upload", "benchmark_signature":sig}

    conn = get_conn()
    try:
        for name, rec, mrr, ndcg in [("BM25",bm25_rec,bm25_mrr,bm25_ndcg),
                                      ("TFIDF",tfidf_rec,tfidf_mrr,tfidf_ndcg),
                                      ("HYBRID",hybrid_rec,hybrid_mrr,hybrid_ndcg)]:
            rid = new_run_id()
            cfg = {**cfg_base, "retriever": name}
            conn.execute("INSERT INTO runs(run_id,created_at,name,notes,config_json) VALUES(?,?,?,?,?)",
                (rid, datetime.utcnow().isoformat()+"Z", f"{name.lower()}_upload", f"{name} on {file.filename}", json.dumps(cfg)))
            for mn, mv in [(f"Recall@{top_k}",rec),(f"MRR@{k_rank}",mrr),(f"nDCG@{k_rank}",ndcg)]:
                conn.execute("INSERT INTO metrics(run_id,metric_name,metric_value,meta_json) VALUES(?,?,?,?)", (rid,mn,mv,None))
        conn.commit()
    finally:
        conn.close()

    # ── Results page ──────────────────────────────────────────
    def row(name, rec, mrr, ndcg, color, is_win):
        bg = "background:rgba(139,92,246,0.08);" if is_win else ""
        badge = "<span style='font-size:11px;color:#a78bfa;margin-left:6px;'>← winner</span>" if is_win else ""
        return f"""<tr style="{bg}"><td style="padding:14px 20px;font-weight:700">{name}{badge}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{rec}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{mrr}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{ndcg}</td></tr>"""

    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench — Upload Results</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3;min-height:100vh}}
nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
.logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
.logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
.nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}.nav-links a:hover{{background:#161b22;color:#e6edf3}}
.cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
.page{{max-width:900px;margin:0 auto;padding:40px}}
.tag{{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:99px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#6ee7b7;font-size:13px;font-weight:500;margin-bottom:14px}}
h1{{font-size:28px;font-weight:800;margin-bottom:6px}}.sub{{font-size:14px;color:#8b949e;margin-bottom:28px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.stat{{background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:16px;text-align:center}}
.stat-val{{font-size:22px;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.stat-label{{font-size:11px;color:#8b949e;margin-top:4px;text-transform:uppercase;letter-spacing:0.4px}}
.winner-bar{{background:linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15));border:1px solid rgba(139,92,246,0.35);border-radius:14px;padding:18px 24px;display:flex;align-items:center;gap:14px;margin-bottom:24px}}
.winner-bar h3{{font-size:16px;font-weight:700;margin-bottom:2px}}.winner-bar p{{font-size:13px;color:#8b949e}}
.winner-chip{{margin-left:auto;padding:8px 18px;background:linear-gradient(135deg,#8b5cf6,#3b82f6);border-radius:99px;font-size:14px;font-weight:700;color:white;white-space:nowrap}}
.card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;overflow:hidden;margin-bottom:20px}}
.card-header{{padding:18px 20px;border-bottom:1px solid #21262d;display:flex;justify-content:space-between}}
.card-title{{font-size:16px;font-weight:700}}.card-sub{{font-size:13px;color:#8b949e}}
table{{width:100%;border-collapse:collapse}}th{{padding:12px 20px;text-align:left;font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;background:rgba(255,255,255,0.02);border-bottom:1px solid #21262d}}
td{{border-bottom:1px solid #21262d;font-size:14px}}tr:last-child td{{border-bottom:none}}
.actions{{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}}.btn{{padding:12px 22px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none}}
.btn-primary{{background:#3b82f6;color:white}}.btn-secondary{{background:#161b22;border:1px solid #21262d;color:#e6edf3}}</style>
</head><body>
<nav><a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
<div class="nav-links"><a href="/dashboard">Dashboard</a><a href="/runs">Runs</a><a href="/upload">Upload</a><a href="/regression">Regression Guard</a></div>
<a href="/upload" class="cta">📄 Test Another</a></nav>
<div class="page">
  <div class="tag">✅ Benchmark Complete</div>
  <h1>Results: {file.filename}</h1>
  <div class="sub">{len(chunks)} chunks · {len(bench)} queries · 3 retrievers · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
  <div class="stats">
    <div class="stat"><div class="stat-val">{len(chunks)}</div><div class="stat-label">Chunks</div></div>
    <div class="stat"><div class="stat-val">{len(bench)}</div><div class="stat-label">Queries</div></div>
    <div class="stat"><div class="stat-val">{chunk_size}</div><div class="stat-label">Chunk Size</div></div>
    <div class="stat"><div class="stat-val">3</div><div class="stat-label">Retrievers</div></div>
  </div>
  <div class="winner-bar"><div style="font-size:28px">🏆</div>
    <div><h3>{winner} wins on {file.filename}</h3><p>MRR@{k_rank} = {scores[winner]:.4f} across {len(bench)} benchmark queries</p></div>
    <div class="winner-chip">{winner} · MRR@{k_rank} = {scores[winner]:.4f}</div></div>
  <div class="card">
    <div class="card-header"><div class="card-title">Full Metric Comparison</div><div class="card-sub">Recall@{top_k} · MRR@{k_rank} · nDCG@{k_rank}</div></div>
    <table><thead><tr><th>Retriever</th><th>Recall@{top_k}</th><th>MRR@{k_rank}</th><th>nDCG@{k_rank}</th></tr></thead>
    <tbody>{row("HYBRID",hybrid_rec,hybrid_mrr,hybrid_ndcg,"#a78bfa",winner=="HYBRID")}
           {row("TFIDF", tfidf_rec, tfidf_mrr, tfidf_ndcg, "#60a5fa",winner=="TFIDF")}
           {row("BM25",  bm25_rec,  bm25_mrr,  bm25_ndcg,  "#34d399",winner=="BM25")}</tbody></table></div>
  <div class="actions">
    <a href="/dashboard" class="btn btn-primary">📊 Full Dashboard</a>
    <a href="/upload" class="btn btn-secondary">📄 Test Another Doc</a>
    <a href="/runs" class="btn btn-secondary">📋 All Runs</a>
    <a href="/compare" class="btn btn-secondary">📈 Trend Chart</a>
  </div>
</div></body></html>""")
