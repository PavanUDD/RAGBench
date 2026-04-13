from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.db.database import init_db, get_conn
from app.core.run_id import new_run_id
from app.reports.report import MetricPoint, build_dashboard_html

from app.core.retrieval import BM25Retriever
from app.core.tfidf import TfidfRetriever
from app.core.hybrid import HybridRetriever
from app.core.ingest import ingest_folder
from app.core.benchmarks import build_benchmark_from_docs
from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k

app = FastAPI(title="RAGBench", version="0.2.0")
templates = Jinja2Templates(directory="app/templates")

# ── Global model cache — loads ONCE at startup ──
_cache = {}


def _get_total_runs():
    try:
        conn = get_conn()
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def detect_regression(conn, metric_name: str = "MRR@10", min_history: int = 3,
                      tolerance: float = 0.02, retriever_filter: str = "HYBRID"):
    rows = conn.execute(
        """
        SELECT r.run_id, r.created_at, r.config_json, m.metric_value
        FROM runs r
        JOIN metrics m ON r.run_id = m.run_id
        WHERE m.metric_name = ?
        ORDER BY r.created_at DESC
        LIMIT 50
        """,
        (metric_name,),
    ).fetchall()

    filtered = []
    for row in rows:
        try:
            cfg = json.loads(row["config_json"]) if row["config_json"] else {}
            if cfg.get("retriever") == retriever_filter:
                filtered.append(row)
        except Exception:
            continue

    rows = filtered
    if not rows:
        return {"status": "insufficient_data"}

    try:
        latest_cfg = json.loads(rows[0]["config_json"]) if rows[0]["config_json"] else {}
        latest_sig = latest_cfg.get("benchmark_signature", None)
    except Exception:
        latest_sig = None

    if latest_sig is not None:
        same = []
        for row in rows:
            try:
                cfg = json.loads(row["config_json"]) if row["config_json"] else {}
                if cfg.get("benchmark_signature") == latest_sig:
                    same.append(row)
            except Exception:
                continue
        rows = same

    if len(rows) < min_history:
        return {"status": "insufficient_data", "retriever": retriever_filter}

    latest = rows[0]
    best = max(rows[1:], key=lambda x: float(x["metric_value"]))
    latest_val = float(latest["metric_value"])
    best_val = float(best["metric_value"])

    if latest_val < (best_val - tolerance):
        return {
            "status": "regression",
            "retriever": retriever_filter,
            "latest_run": latest["run_id"],
            "latest_value": latest_val,
            "best_run": best["run_id"],
            "best_value": best_val,
        }
    return {
        "status": "ok",
        "retriever": retriever_filter,
        "latest_run": latest["run_id"],
        "latest_value": latest_val,
        "best_value": best_val,
    }


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def _startup():
    init_db()
    print("[RAGBench] 🚀 Pre-loading Dense model at startup...")
    t0 = time.time()
    chunks = ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    chunk_map = {c.chunk_id: c.text for c in chunks}
    _cache["hybrid"] = HybridRetriever(chunk_map)
    _cache["chunks"] = chunks
    _cache["chunk_map"] = chunk_map
    elapsed = round((time.time() - t0) * 1000)
    print(f"[RAGBench] ✅ Model ready in {elapsed}ms — fast from here on")


# ── HOME ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    home_path = Path("app/templates/home.html")
    if home_path.exists():
        return HTMLResponse(home_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>RAGBench ✅</h2><a href='/demo-run'>Run Benchmark</a>")


# ── DASHBOARD ────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    chunk_map = _cache.get("chunk_map") or {
        c.chunk_id: c.text
        for c in ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    }
    chunks = _cache.get("chunks") or ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    bench = build_benchmark_from_docs(chunks)
    k_recall, k_rank = 5, 10

    def eval_with_latency(retriever):
        recalls, mrrs, ndcgs, latencies = [], [], [], []
        for bq in bench:
            t0 = time.time()
            results = retriever.search(bq.query, k=k_rank)
            latencies.append((time.time() - t0) * 1000)
            ids = [r.doc_id for r in results]
            rel = set(bq.relevant_chunk_ids)
            recalls.append(recall_at_k(rel, ids, k=k_recall))
            mrrs.append(mrr_at_k(rel, ids, k=k_rank))
            ndcgs.append(ndcg_at_k(rel, ids, k=k_rank))
        avg = lambda xs: round(sum(xs) / max(len(xs), 1), 4)
        return {
            "recall5": avg(recalls),
            "mrr10": avg(mrrs),
            "ndcg10": avg(ndcgs),
            "latency_ms": round(sum(latencies) / len(latencies), 1),
        }

    hybrid_ret = _cache.get("hybrid") or HybridRetriever(chunk_map)
    bm25_d   = {"name": "BM25",   **eval_with_latency(BM25Retriever(chunk_map))}
    tfidf_d  = {"name": "TFIDF",  **eval_with_latency(TfidfRetriever(chunk_map))}
    hybrid_d = {"name": "HYBRID", **eval_with_latency(hybrid_ret)}

    all_scores = [bm25_d, tfidf_d, hybrid_d]
    winner_data = max(all_scores, key=lambda x: x["mrr10"])
    winner = winner_data["name"]
    bm25_baseline = bm25_d["mrr10"]

    def make_ctx(d):
        name = d["name"]
        is_winner = name == winner
        cfg = {
            "HYBRID": ("#a78bfa", "bar-purple", "tag-winner" if is_winner else "tag-strong",
                       "🏆 Winner" if is_winner else "Hybrid", "BM25 + Dense Embeddings (RRF)"),
            "TFIDF":  ("#60a5fa", "bar-blue",   "tag-winner" if is_winner else "tag-strong",
                       "🏆 Winner" if is_winner else "Strong",  "TF-IDF Cosine Similarity"),
            "BM25":   ("#34d399", "bar-green",  "tag-winner" if is_winner else "tag-baseline",
                       "🏆 Winner" if is_winner else "Baseline", "BM25 Okapi Lexical"),
        }[name]
        color, bar_class, tag_class, tag_label, method = cfg
        score_class = "score-good" if is_winner else ("score-ok" if name == "TFIDF" else "score-base")
        delta = round(d["mrr10"] - bm25_baseline, 4)
        return {
            **d,
            "is_winner": is_winner,
            "color": color,
            "bar_class": bar_class,
            "tag_class": tag_class,
            "tag_label": tag_label,
            "method": method,
            "score_class": score_class,
            "recall5_pct":  round(d["recall5"] * 100, 1),
            "mrr10_pct":    round(d["mrr10"]   * 100, 1),
            "ndcg10_pct":   round(d["ndcg10"]  * 100, 1),
            "delta_mrr":     delta,
            "delta_mrr_str": f"{abs(delta):.4f}",
        }

    retrievers_ctx = [make_ctx(hybrid_d), make_ctx(tfidf_d), make_ctx(bm25_d)]
    gap = round(winner_data["mrr10"] - bm25_baseline, 3)

    if winner == "HYBRID":
        verdict_why = "Combining BM25 lexical matching with dense semantic embeddings via RRF fusion captures both exact keyword hits and conceptual similarity — neither retriever alone achieves this."
        verdict_rec = "Deploy HYBRID for production. Latency overhead is justified by MRR improvement. Use TFIDF as a lightweight fallback if speed is critical."
    elif winner == "TFIDF":
        verdict_why = "TF-IDF outperforms dense embeddings on this corpus because the documents are keyword-dense. Semantic similarity adds less signal when vocabulary is highly consistent across docs and queries."
        verdict_rec = "Deploy TFIDF for this corpus. Re-evaluate with a larger, more semantically diverse document set — HYBRID typically gains an edge as corpus diversity grows."
    else:
        verdict_why = "BM25 leads on this corpus, indicating high lexical precision between benchmark queries and documents. Ground-truth queries use terminology that directly matches document content."
        verdict_rec = "BM25 is sufficient here. Monitor as the corpus grows — semantic retrievers improve with document diversity and paraphrasing in user queries."

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "winner": winner,
        "winner_mrr": winner_data["mrr10"],
        "verdict_summary": f"{winner} achieves MRR@10 of {winner_data['mrr10']} — outperforming BM25 baseline by {gap} across {len(bench)} benchmark queries and {len(chunk_map)} indexed chunks.",
        "verdict_why": verdict_why,
        "verdict_rec": verdict_rec,
        "bm25_gap": f"{gap:.3f}",
        "retrievers": retrievers_ctx,
        "total_runs": _get_total_runs(),
        "doc_count": len(chunk_map),
        "query_count": len(bench),
    })


# ── DEMO RUN ──────────────────────────────────────────────────
@app.get("/demo-run", response_class=HTMLResponse)
def demo_run():
    chunk_map = _cache.get("chunk_map") or {
        c.chunk_id: c.text
        for c in ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    }
    chunks = _cache.get("chunks") or ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    bench = build_benchmark_from_docs(chunks)
    k_recall, k_rank = 5, 10

    config = {
        "dataset": "docs_folder:data/docs",
        "chunking": {"chunk_size_words": 120, "overlap_words": 25},
        "k_recall": k_recall, "k_rank": k_rank,
        "notes": "Docs-based benchmark run"
    }

    def eval_retriever(retriever):
        recalls, mrrs, ndcgs = [], [], []
        for bq in bench:
            results = retriever.search(bq.query, k=k_rank)
            ids = [r.doc_id for r in results]
            rel = set(bq.relevant_chunk_ids)
            recalls.append(recall_at_k(rel, ids, k=k_recall))
            mrrs.append(mrr_at_k(rel, ids, k=k_rank))
            ndcgs.append(ndcg_at_k(rel, ids, k=k_rank))
        avg = lambda xs: sum(xs) / max(len(xs), 1)
        return [
            MetricPoint(f"Recall@{k_recall}", round(avg(recalls), 4)),
            MetricPoint(f"MRR@{k_rank}",      round(avg(mrrs),    4)),
            MetricPoint(f"nDCG@{k_rank}",     round(avg(ndcgs),   4)),
            MetricPoint("Chunks",  float(len(chunks))),
            MetricPoint("Queries", float(len(bench))),
        ]

    hybrid_ret   = _cache.get("hybrid") or HybridRetriever(chunk_map)
    bm25_metrics   = eval_retriever(BM25Retriever(chunk_map))
    tfidf_metrics  = eval_retriever(TfidfRetriever(chunk_map))
    hybrid_metrics = eval_retriever(hybrid_ret)

    conn = get_conn()
    try:
        sig = {
            "docs_folder": "data/docs",
            "chunks": len(chunks),
            "queries": len(bench),
            "chunking": {"chunk_size": 120, "overlap": 25},
        }

        def save_one(name, metrics_list):
            rid = new_run_id()
            cfg = {**config, "retriever": name, "benchmark_signature": sig}
            conn.execute(
                "INSERT INTO runs(run_id, created_at, name, notes, config_json) VALUES(?,?,?,?,?)",
                (rid, datetime.utcnow().isoformat() + "Z",
                 f"{name.lower()}_run", f"{name} benchmark run", json.dumps(cfg)),
            )
            for m in metrics_list:
                if m.name in ("Chunks", "Queries"):
                    continue
                conn.execute(
                    "INSERT INTO metrics(run_id, metric_name, metric_value, meta_json) VALUES(?,?,?,?)",
                    (rid, m.name, float(m.value), None),
                )
            return rid

        run_id_bm25   = save_one("BM25",   bm25_metrics)
        run_id_tfidf  = save_one("TFIDF",  tfidf_metrics)
        run_id_hybrid = save_one("HYBRID", hybrid_metrics)
        conn.commit()
    finally:
        conn.close()

    # Build a clean 3-retriever results page
    def score(metrics, name):
        return {m.name: m.value for m in metrics if m.name not in ("Chunks","Queries")}

    bm25_s   = score(bm25_metrics,   "BM25")
    tfidf_s  = score(tfidf_metrics,  "TFIDF")
    hybrid_s = score(hybrid_metrics, "HYBRID")

    # Save BM25 report to disk
    html_report = build_dashboard_html(
        run_id=run_id_bm25,
        title="BM25 Metrics",
        metrics=[m for m in bm25_metrics if m.name not in ("Chunks","Queries")]
    )
    out_dir = Path("runs") / run_id_bm25
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html_report, encoding="utf-8")

    def row(name, s, color, highlight=False):
        bg = "background:rgba(139,92,246,0.08);" if highlight else ""
        badge = "<span style='font-size:11px;color:#a78bfa;margin-left:6px;'>← winner</span>" if highlight else ""
        return f"""
        <tr style="{bg}">
          <td style="padding:14px 20px;font-weight:700;font-size:15px;">{name}{badge}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color};">{s.get(f'Recall@{k_recall}','—')}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color};">{s.get(f'MRR@{k_rank}','—')}</td>
          <td style="padding:14px 20px;font-weight:700;color:{color};">{s.get(f'nDCG@{k_rank}','—')}</td>
        </tr>"""

    scores_map = {
        "HYBRID": hybrid_s.get(f"MRR@{k_rank}", 0),
        "TFIDF":  tfidf_s.get(f"MRR@{k_rank}",  0),
        "BM25":   bm25_s.get(f"MRR@{k_rank}",   0),
    }
    winner = max(scores_map, key=scores_map.get)

    return HTMLResponse(f"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RAGBench — Run Complete</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3;min-height:100vh}}
  nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9);position:sticky;top:0;z-index:100}}
  .logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
  .logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}}
  .nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}
  .nav-links a:hover{{background:#161b22;color:#e6edf3}}
  .cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
  .page{{max-width:900px;margin:0 auto;padding:40px}}
  .header{{margin-bottom:28px}}
  .run-tag{{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:99px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#6ee7b7;font-size:13px;font-weight:500;margin-bottom:14px}}
  h1{{font-size:28px;font-weight:800;letter-spacing:-0.5px;margin-bottom:6px}}
  .sub{{font-size:14px;color:#8b949e}}
  .winner-bar{{background:linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15));border:1px solid rgba(139,92,246,0.35);border-radius:14px;padding:18px 24px;display:flex;align-items:center;gap:14px;margin-bottom:24px}}
  .winner-bar h3{{font-size:16px;font-weight:700;margin-bottom:2px}}
  .winner-bar p{{font-size:13px;color:#8b949e}}
  .winner-chip{{margin-left:auto;padding:8px 18px;background:linear-gradient(135deg,#8b5cf6,#3b82f6);border-radius:99px;font-size:14px;font-weight:700;color:white;white-space:nowrap}}
  .card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;overflow:hidden;margin-bottom:20px}}
  .card-header{{padding:18px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;justify-content:space-between}}
  .card-title{{font-size:16px;font-weight:700}}
  .card-sub{{font-size:13px;color:#8b949e}}
  table{{width:100%;border-collapse:collapse}}
  th{{padding:12px 20px;text-align:left;font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;background:rgba(255,255,255,0.02);border-bottom:1px solid #21262d}}
  td{{border-bottom:1px solid #21262d;font-size:14px}}
  tr:last-child td{{border-bottom:none}}
  .actions{{display:flex;gap:12px;margin-top:24px}}
  .btn{{padding:12px 22px;border-radius:10px;font-size:14px;font-weight:600;text-decoration:none;transition:all 0.15s}}
  .btn-primary{{background:#3b82f6;color:white}}
  .btn-primary:hover{{background:#2563eb;transform:translateY(-1px)}}
  .btn-secondary{{background:#161b22;border:1px solid #21262d;color:#e6edf3}}
  .btn-secondary:hover{{border-color:#3b82f6;transform:translateY(-1px)}}
  .meta{{font-size:12px;color:#6e7681;margin-top:16px}}
</style>
</head><body>
<nav>
  <a href="/" class="logo"><div class="logo-icon">⚡</div>RAGBench</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a><a href="/runs">Runs</a>
    <a href="/compare">Compare</a><a href="/regression">Regression Guard</a>
  </div>
  <a href="/demo-run" class="cta">▶ New Run</a>
</nav>
<div class="page">
  <div class="header">
    <div class="run-tag">✅ Run Complete</div>
    <h1>Benchmark Results</h1>
    <div class="sub">3 retrievers evaluated · {len(bench)} queries · {len(chunk_map)} chunks · Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
  </div>
  <div class="winner-bar">
    <div style="font-size:28px">🏆</div>
    <div>
      <h3>{winner} wins this benchmark</h3>
      <p>Highest MRR@{k_rank} = {scores_map[winner]:.4f} across {len(bench)} ground-truth queries</p>
    </div>
    <div class="winner-chip">{winner} · MRR@{k_rank} = {scores_map[winner]:.4f}</div>
  </div>
  <div class="card">
    <div class="card-header">
      <div class="card-title">Full Metric Comparison</div>
      <div class="card-sub">Recall@{k_recall} · MRR@{k_rank} · nDCG@{k_rank}</div>
    </div>
    <table>
      <thead><tr><th>Retriever</th><th>Recall@{k_recall}</th><th>MRR@{k_rank}</th><th>nDCG@{k_rank}</th></tr></thead>
      <tbody>
        {row("HYBRID", hybrid_s, "#a78bfa", winner=="HYBRID")}
        {row("TFIDF",  tfidf_s,  "#60a5fa", winner=="TFIDF")}
        {row("BM25",   bm25_s,   "#34d399", winner=="BM25")}
      </tbody>
    </table>
  </div>
  <div class="actions">
    <a href="/dashboard" class="btn btn-primary">📊 Open Full Dashboard</a>
    <a href="/runs" class="btn btn-secondary">📋 View All Runs</a>
    <a href="/analysis/{run_id_bm25}" class="btn btn-secondary">🔬 Failure Analysis</a>
    <a href="/compare" class="btn btn-secondary">📈 Trend Chart</a>
  </div>
  <div class="meta">Run IDs: BM25={run_id_bm25} · TFIDF={run_id_tfidf} · HYBRID={run_id_hybrid}</div>
</div>
</body></html>
""")


# ── ANALYSIS ─────────────────────────────────────────────────
@app.get("/analysis/{run_id}", response_class=HTMLResponse)
def analysis(run_id: str, request: Request, q: int = 0):
    chunk_map = _cache.get("chunk_map") or {
        c.chunk_id: c.text
        for c in ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    }
    chunks = _cache.get("chunks") or ingest_folder(folder="data/docs", chunk_size=120, overlap=25)

    DOC_LABELS = {
        "api_errors_legacy": "LEGACY/OUTDATED",
        "legacy_logging_guidelines": "LEGACY/OUTDATED",
        "logging_minimal_legacy": "LEGACY/OUTDATED",
        "incident_calendar_policy": "POLICY (NOT INCIDENT STEPS)",
        "security_logging_exceptions": "EXCEPTION POLICY",
    }

    retriever = BM25Retriever(chunk_map)
    bench = build_benchmark_from_docs(chunks)

    if not bench:
        return HTMLResponse("<h3>No benchmark queries found.</h3>")

    selected_idx = max(0, min(int(q), len(bench) - 1))
    selected = bench[selected_idx]
    k_rank = 10
    results = retriever.search(selected.query, k=k_rank)
    relevant = set(selected.relevant_chunk_ids)
    relevant_ids = ", ".join(sorted(list(relevant))) if relevant else "(none)"

    rows = []
    hit = False
    hit_rank = None
    for i, r in enumerate(results, start=1):
        is_rel = r.doc_id in relevant
        if is_rel and not hit:
            hit = True
            hit_rank = i
        txt = chunk_map.get(r.doc_id, "")
        preview = txt[:220] + ("..." if len(txt) > 220 else "")
        doc_id = r.doc_id.split("::")[0]
        label = DOC_LABELS.get(doc_id, "CURRENT")
        rows.append({
            "rank": i, "chunk_id": r.doc_id, "score": r.score,
            "preview": preview, "is_relevant": is_rel, "label": label
        })

    why = ""
    if not hit and relevant:
        rel_id = list(relevant)[0]
        rel_text = chunk_map.get(rel_id, "")
        q_terms = set(t.lower().strip(".,!?;:()[]{}\"'") for t in selected.query.split() if t.strip())
        d_terms = set(t.lower().strip(".,!?;:()[]{}\"'") for t in rel_text.split() if t.strip())
        overlap = sorted(q_terms.intersection(d_terms))
        if not overlap:
            why = "Low lexical overlap: query terms do not appear in the relevant chunk. BM25 relies on term matches — try synonyms, smaller chunks, or hybrid retrieval."
        else:
            why = f"Some overlap exists ({', '.join(overlap)}), but other chunks scored higher. Try smaller chunks or more distinctive terms."

    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "run_id": run_id,
        "queries": bench,
        "selected_idx": selected_idx,
        "selected": selected,
        "k_rank": k_rank,
        "relevant_count": len(relevant),
        "relevant_ids": relevant_ids,
        "retrieved": rows,
        "hit": hit,
        "hit_rank": hit_rank,
        "why": why,
    })


# ── RUNS ─────────────────────────────────────────────────────
@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT run_id, created_at FROM runs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        out = []
        for row in rows:
            run_id = row["run_id"]
            metrics = conn.execute(
                "SELECT metric_name, metric_value FROM metrics WHERE run_id = ? ORDER BY metric_name",
                (run_id,),
            ).fetchall()
            cfg_row = conn.execute(
                "SELECT config_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            retriever_name = "unknown"
            if cfg_row and cfg_row["config_json"]:
                try:
                    cfg = json.loads(cfg_row["config_json"])
                    retriever_name = cfg.get("retriever", "unknown")
                except Exception:
                    pass
            out.append({
                "run_id": run_id,
                "created_at": row["created_at"],
                "retriever": retriever_name,
                "metrics": [{"name": str(m["metric_name"]), "value": round(float(m["metric_value"]), 4)} for m in metrics],
            })
    finally:
        conn.close()
    return templates.TemplateResponse("runs.html", {"request": request, "runs": out})


# ── OPEN REPORT ───────────────────────────────────────────────
@app.get("/open-report/{run_id}", response_class=HTMLResponse)
def open_report(run_id: str):
    report_path = Path("runs") / run_id / "report.html"
    if not report_path.exists():
        return HTMLResponse(f"<h3>Report not found for {run_id}</h3>", status_code=404)
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


# ── COMPARE ───────────────────────────────────────────────────
@app.get("/compare", response_class=HTMLResponse)
def compare():
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.templates.default = "plotly_dark"

    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT run_id, created_at, config_json FROM runs ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
        points = []
        for row in rows:
            retriever = "unknown"
            if row["config_json"]:
                try:
                    retriever = json.loads(row["config_json"]).get("retriever", "unknown")
                except Exception:
                    pass
            mrr = conn.execute(
                "SELECT metric_value FROM metrics WHERE run_id = ? AND metric_name = 'MRR@10'",
                (row["run_id"],),
            ).fetchone()
            if mrr:
                points.append((row["created_at"], retriever, float(mrr["metric_value"]), row["run_id"]))
    finally:
        conn.close()

    by_ret = {}
    for created_at, retriever, mrr, run_id in points:
        by_ret.setdefault(retriever, []).append((created_at, mrr, run_id))

    colors = {"HYBRID": "#a78bfa", "TFIDF": "#60a5fa", "BM25": "#34d399"}
    fig = go.Figure()
    for retriever, xs in by_ret.items():
        xs_sorted = sorted(xs, key=lambda x: x[0])
        fig.add_trace(go.Scatter(
            x=[a for a, _, __ in xs_sorted],
            y=[b for _, b, __ in xs_sorted],
            mode="lines+markers",
            name=retriever,
            line=dict(color=colors.get(retriever, "#ffffff"), width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title="MRR@10 by Retriever (Recent Runs)",
        xaxis_title="Timestamp (UTC)", yaxis_title="MRR@10",
        height=520, margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return HTMLResponse(f"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench Compare</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3}}
  nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
  .logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
  .logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
  .nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}
  .nav-links a:hover{{background:#161b22;color:#e6edf3}}
  .cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
  .page{{max-width:1200px;margin:0 auto;padding:40px}}
  h1{{font-size:24px;font-weight:800;margin-bottom:6px}}
  .sub{{font-size:14px;color:#8b949e;margin-bottom:28px}}
  .card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:20px}}
</style>
</head><body>
<nav>
  <a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a><a href="/runs">Runs</a>
    <a href="/compare">Compare</a><a href="/regression">Regression Guard</a>
  </div>
  <a href="/demo-run" class="cta">▶ New Run</a>
</nav>
<div class="page">
  <h1>Retriever Comparison</h1>
  <div class="sub">MRR@10 trend over time · BM25 vs TF-IDF vs Hybrid Dense+Sparse</div>
  <div class="card">{chart}</div>
</div>
</body></html>
""")


# ── REGRESSION GUARD ──────────────────────────────────────────
@app.get("/regression", response_class=HTMLResponse)
def regression():
    conn = get_conn()
    try:
        result = detect_regression(conn, retriever_filter="HYBRID")
    finally:
        conn.close()

    if result["status"] == "insufficient_data":
        status_html = """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <span style="font-size:24px">⏳</span>
          <span style="font-size:18px;font-weight:700;color:#f59e0b">Collecting Data</span>
        </div>
        <p style="color:#8b949e">Run at least 3 benchmarks to enable regression detection.</p>"""
    elif result["status"] == "regression":
        status_html = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <span style="font-size:24px">🚨</span>
          <span style="font-size:18px;font-weight:700;color:#ef4444">REGRESSION DETECTED</span>
        </div>
        <p style="color:#8b949e;margin-bottom:16px">Latest HYBRID run shows degraded retrieval quality.</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:400px;">
          <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;">
            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Latest Run MRR@10</div>
            <div style="font-size:22px;font-weight:800;color:#ef4444">{result['latest_value']:.4f}</div>
          </div>
          <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;">
            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Best Run MRR@10</div>
            <div style="font-size:22px;font-weight:800;color:#10b981">{result['best_value']:.4f}</div>
          </div>
        </div>"""
    else:
        status_html = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <span style="font-size:24px">✅</span>
          <span style="font-size:18px;font-weight:700;color:#10b981">No Regression Detected</span>
        </div>
        <p style="color:#8b949e;margin-bottom:16px">HYBRID retrieval quality is stable.</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:400px;">
          <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;">
            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Latest MRR@10</div>
            <div style="font-size:22px;font-weight:800;color:#a78bfa">{result['latest_value']:.4f}</div>
          </div>
          <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px;">
            <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Best MRR@10</div>
            <div style="font-size:22px;font-weight:800;color:#10b981">{result['best_value']:.4f}</div>
          </div>
        </div>"""

    return HTMLResponse(f"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench — Regression Guard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3}}
  nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
  .logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
  .logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
  .nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}
  .nav-links a:hover{{background:#161b22;color:#e6edf3}}
  .cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
  .page{{max-width:900px;margin:0 auto;padding:40px}}
  h1{{font-size:24px;font-weight:800;margin-bottom:6px}}
  .sub{{font-size:14px;color:#8b949e;margin-bottom:32px}}
  .card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:28px}}
  .back{{display:inline-flex;align-items:center;gap:6px;margin-top:24px;color:#3b82f6;text-decoration:none;font-size:14px;font-weight:500}}
</style>
</head><body>
<nav>
  <a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a><a href="/runs">Runs</a>
    <a href="/compare">Compare</a><a href="/regression">Regression Guard</a>
  </div>
  <a href="/demo-run" class="cta">▶ New Run</a>
</nav>
<div class="page">
  <h1>Regression Guard</h1>
  <div class="sub">Monitors HYBRID retriever MRR@10 — alerts when quality drops below best recent run</div>
  <div class="card">
    {status_html}
    <a href="/runs" class="back">← Back to Runs</a>
  </div>
</div>
</body></html>
""")