from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.db.database import get_conn
from app.core.run_id import new_run_id
from app.core.retrieval import BM25Retriever
from app.core.tfidf import TfidfRetriever
from app.core.benchmarks import build_benchmark_from_docs
from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from app.core.ingest import ingest_folder
from app.reports.report import MetricPoint, build_dashboard_html

router = APIRouter()

def _get_cache():
    from app.main import _cache
    return _cache

def _eval(retriever, bench, k_recall=5, k_rank=10):
    recalls, mrrs, ndcgs = [], [], []
    for bq in bench:
        ids = [r.doc_id for r in retriever.search(bq.query, k=k_rank)]
        rel = set(bq.relevant_chunk_ids)
        recalls.append(recall_at_k(rel, ids, k=k_recall))
        mrrs.append(mrr_at_k(rel, ids, k=k_rank))
        ndcgs.append(ndcg_at_k(rel, ids, k=k_rank))
    avg = lambda xs: round(sum(xs) / max(len(xs), 1), 4)
    return [
        MetricPoint(f"Recall@{k_recall}", avg(recalls)),
        MetricPoint(f"MRR@{k_rank}",      avg(mrrs)),
        MetricPoint(f"nDCG@{k_rank}",     avg(ndcgs)),
    ]


@router.get("/demo-run", response_class=HTMLResponse)
def demo_run():
    cache = _get_cache()
    chunks   = cache.get("chunks")   or ingest_folder("data/docs", chunk_size=120, overlap=25)
    chunk_map = cache.get("chunk_map") or {c.chunk_id: c.text for c in chunks}
    hybrid   = cache.get("hybrid")

    bench = build_benchmark_from_docs(chunks)
    k_recall, k_rank = 5, 10

    bm25_m   = _eval(BM25Retriever(chunk_map),  bench)
    tfidf_m  = _eval(TfidfRetriever(chunk_map), bench)
    hybrid_m = _eval(hybrid, bench)

    sig = {"docs_folder":"data/docs","chunks":len(chunks),"queries":len(bench),"chunking":{"chunk_size":120,"overlap":25}}
    config_base = {"dataset":"docs_folder:data/docs","chunking":{"chunk_size_words":120,"overlap_words":25},"k_recall":k_recall,"k_rank":k_rank}

    conn = get_conn()
    try:
        ids = {}
        for name, metrics in [("BM25", bm25_m), ("TFIDF", tfidf_m), ("HYBRID", hybrid_m)]:
            rid = new_run_id()
            ids[name] = rid
            cfg = {**config_base, "retriever": name, "benchmark_signature": sig}
            conn.execute("INSERT INTO runs(run_id,created_at,name,notes,config_json) VALUES(?,?,?,?,?)",
                (rid, datetime.utcnow().isoformat()+"Z", f"{name.lower()}_run", f"{name} benchmark", json.dumps(cfg)))
            for m in metrics:
                conn.execute("INSERT INTO metrics(run_id,metric_name,metric_value,meta_json) VALUES(?,?,?,?)",
                    (rid, m.name, float(m.value), None))
        conn.commit()
    finally:
        conn.close()

    def s(metrics): return {m.name: m.value for m in metrics}
    bm25_s, tfidf_s, hybrid_s = s(bm25_m), s(tfidf_m), s(hybrid_m)
    scores = {"HYBRID": hybrid_s[f"MRR@{k_rank}"], "TFIDF": tfidf_s[f"MRR@{k_rank}"], "BM25": bm25_s[f"MRR@{k_rank}"]}
    winner = max(scores, key=scores.get)

    html_report = build_dashboard_html(ids["BM25"], "BM25 Metrics", bm25_m)
    out_dir = Path("runs") / ids["BM25"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html_report, encoding="utf-8")

    def row(name, sc, color, is_win):
        bg = "background:rgba(139,92,246,0.08);" if is_win else ""
        badge = "<span style='font-size:11px;color:#a78bfa;margin-left:6px;'>← winner</span>" if is_win else ""
        return f"""<tr style="{bg}">
          <td style="padding:14px 20px;font-weight:700">{name}{badge}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{sc.get(f'Recall@{k_recall}','—')}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{sc.get(f'MRR@{k_rank}','—')}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color}">{sc.get(f'nDCG@{k_rank}','—')}</td>
        </tr>"""

    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench — Run Complete</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3;min-height:100vh}}
nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
.logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
.logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
.nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}.nav-links a:hover{{background:#161b22;color:#e6edf3}}
.cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
.page{{max-width:900px;margin:0 auto;padding:40px}}
.tag{{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:99px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#6ee7b7;font-size:13px;font-weight:500;margin-bottom:14px}}
h1{{font-size:28px;font-weight:800;letter-spacing:-0.5px;margin-bottom:6px}}.sub{{font-size:14px;color:#8b949e;margin-bottom:28px}}
.winner-bar{{background:linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15));border:1px solid rgba(139,92,246,0.35);border-radius:14px;padding:18px 24px;display:flex;align-items:center;gap:14px;margin-bottom:24px}}
.winner-bar h3{{font-size:16px;font-weight:700;margin-bottom:2px}}.winner-bar p{{font-size:13px;color:#8b949e}}
.winner-chip{{margin-left:auto;padding:8px 18px;background:linear-gradient(135deg,#8b5cf6,#3b82f6);border-radius:99px;font-size:14px;font-weight:700;color:white;white-space:nowrap}}
.card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;overflow:hidden;margin-bottom:20px}}
.card-header{{padding:18px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;justify-content:space-between}}
.card-title{{font-size:16px;font-weight:700}}.card-sub{{font-size:13px;color:#8b949e}}
table{{width:100%;border-collapse:collapse}}th{{padding:12px 20px;text-align:left;font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;background:rgba(255,255,255,0.02);border-bottom:1px solid #21262d}}
td{{border-bottom:1px solid #21262d;font-size:14px}}tr:last-child td{{border-bottom:none}}
.actions{{display:flex;gap:12px;margin-top:24px;flex-wrap:wrap}}.btn{{padding:12px 22px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none}}
.btn-primary{{background:#3b82f6;color:white}}.btn-secondary{{background:#161b22;border:1px solid #21262d;color:#e6edf3}}</style>
</head><body>
<nav><a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
<div class="nav-links"><a href="/dashboard">Dashboard</a><a href="/runs">Runs</a><a href="/compare">Compare</a><a href="/upload">Upload Doc</a></div>
<a href="/demo-run" class="cta">▶ New Run</a></nav>
<div class="page">
  <div class="tag">✅ Run Complete</div>
  <h1>Benchmark Results</h1>
  <div class="sub">3 retrievers · {len(bench)} queries · {len(chunk_map)} chunks · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
  <div class="winner-bar"><div style="font-size:28px">🏆</div>
    <div><h3>{winner} wins this benchmark</h3><p>MRR@{k_rank} = {scores[winner]:.4f} across {len(bench)} ground-truth queries</p></div>
    <div class="winner-chip">{winner} · MRR@{k_rank} = {scores[winner]:.4f}</div></div>
  <div class="card">
    <div class="card-header"><div class="card-title">Full Metric Comparison</div><div class="card-sub">Recall@{k_recall} · MRR@{k_rank} · nDCG@{k_rank}</div></div>
    <table><thead><tr><th>Retriever</th><th>Recall@{k_recall}</th><th>MRR@{k_rank}</th><th>nDCG@{k_rank}</th></tr></thead>
    <tbody>{row("HYBRID",hybrid_s,"#a78bfa",winner=="HYBRID")}{row("TFIDF",tfidf_s,"#60a5fa",winner=="TFIDF")}{row("BM25",bm25_s,"#34d399",winner=="BM25")}</tbody></table></div>
  <div class="actions">
    <a href="/dashboard" class="btn btn-primary">📊 Full Dashboard</a>
    <a href="/runs" class="btn btn-secondary">📋 All Runs</a>
    <a href="/analysis/{ids['BM25']}" class="btn btn-secondary">🔬 Failure Analysis</a>
    <a href="/upload" class="btn btn-secondary">📄 Test Your Doc</a>
  </div>
</div></body></html>""")
