from __future__ import annotations

import json
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

app = FastAPI(title="RAGBench", version="0.1.0")
templates = Jinja2Templates(directory="app/templates")

def detect_regression(conn, metric_name: str = "MRR@10", min_history: int = 3, tolerance: float = 0.02, retriever_filter: str = "TFIDF"):
    """
    Detects whether the latest run regressed compared to the best recent run.
    Default: compares TFIDF runs only (stable).
    """
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

    # Filter to a specific retriever (default TFIDF) to avoid mixing apples/oranges
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

    # Determine signature from latest run
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



# Optional static folder if we add assets later
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family: system-ui; padding: 24px;">
      <h2>RAGBench is running ✅</h2>
      <ul>
        <li><a href="/demo-run">/demo-run</a> — generate a new benchmark run</li>
        <li><a href="/runs">/runs</a> — view all runs (leaderboard)</li>
        <li><a href="/compare">/compare</a> — retriever comparison chart</li>
        <li><a href="/regression">/regression</a> — regression guard (prevent bad releases)</li>
      </ul>
    </body></html>
    """


@app.get("/demo-run", response_class=HTMLResponse)
def demo_run():
    run_id = new_run_id()
    created_at = datetime.utcnow().isoformat() + "Z"

   
    config = {
        "dataset": "docs_folder:data/docs",
        "chunking": {"chunk_size_words": 120, "overlap_words": 25},
        "k_recall": 5,
        "k_rank": 10,
        "notes": "Docs-based benchmark run"
    }

    # Demo metrics (we'll replace with real Recall@k/MRR soon)
    # REAL evaluation: load dataset -> run BM25 -> compute metrics
    from app.core.dataset import load_dataset
    from app.core.retrieval import BM25Retriever
    from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k

        # REAL docs ingestion -> chunking -> BM25 over chunks -> benchmark queries
    from app.core.ingest import ingest_folder
    from app.core.benchmarks import build_benchmark_from_docs
    from app.core.retrieval import BM25Retriever
    from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k

    from app.core.ingest import ingest_folder
    from app.core.benchmarks import build_benchmark_from_docs
    from app.core.metrics import recall_at_k, mrr_at_k, ndcg_at_k
    from app.core.retrieval import BM25Retriever
    from app.core.tfidf import TfidfRetriever

    chunks = ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    chunk_map = {c.chunk_id: c.text for c in chunks}
    bench = build_benchmark_from_docs(chunks)

    k_recall = 5
    k_rank = 10

    def eval_retriever(retriever, name: str):
        recalls, mrrs, ndcgs = [], [], []
        for bq in bench:
            results = retriever.search(bq.query, k=k_rank)
            retrieved_ids = [r.doc_id for r in results]
            relevant = set(bq.relevant_chunk_ids)
            recalls.append(recall_at_k(relevant, retrieved_ids, k=k_recall))
            mrrs.append(mrr_at_k(relevant, retrieved_ids, k=k_rank))
            ndcgs.append(ndcg_at_k(relevant, retrieved_ids, k=k_rank))

        def avg(xs): return sum(xs) / max(len(xs), 1)

        return [
            MetricPoint("Retriever", 0.0),  # placeholder for display (we'll render name elsewhere later)
            MetricPoint(f"Recall@{k_recall}", round(avg(recalls), 4)),
            MetricPoint(f"MRR@{k_rank}", round(avg(mrrs), 4)),
            MetricPoint(f"nDCG@{k_rank}", round(avg(ndcgs), 4)),
            MetricPoint("Chunks", float(len(chunks))),
            MetricPoint("Queries", float(len(bench))),
        ]

    # Evaluate BM25
    bm25 = BM25Retriever(chunk_map)
    bm25_metrics = eval_retriever(bm25, "BM25")

    # Evaluate TF-IDF
    tfidf = TfidfRetriever(chunk_map)
    tfidf_metrics = eval_retriever(tfidf, "TFIDF")



    # Save run + metrics
    conn = get_conn()
    try:
        def save_one(run_id: str, retriever_name: str, metrics_list):
            created_at = datetime.utcnow().isoformat() + "Z"
            cfg = dict(config)
            # benchmark signature: ensures regressions are compared apples-to-apples
            cfg["benchmark_signature"] = {
                "docs_folder": "data/docs",
                "chunks": len(chunks),
                "queries": len(bench),
                "chunking": {"chunk_size": 120, "overlap": 25},
            }

            cfg["retriever"] = retriever_name

            conn.execute(
                "INSERT INTO runs(run_id, created_at, name, notes, config_json) VALUES(?,?,?,?,?)",
                (run_id, created_at, f"{retriever_name.lower()}_run", f"{retriever_name} benchmark run", json.dumps(cfg)),
            )
            for m in metrics_list:
                # skip placeholder
                if m.name == "Retriever":
                    continue
                conn.execute(
                    "INSERT INTO metrics(run_id, metric_name, metric_value, meta_json) VALUES(?,?,?,?)",
                    (run_id, m.name, float(m.value), None),
                )

        run_id_bm25 = new_run_id()
        run_id_tfidf = new_run_id()

        save_one(run_id_bm25, "BM25", bm25_metrics)
        save_one(run_id_tfidf, "TFIDF", tfidf_metrics)

        conn.commit()
    finally:
        conn.close()


    html = build_dashboard_html(run_id=run_id_bm25, title="BM25 Metrics Dashboard", metrics=bm25_metrics[1:])


    # Also write the report to disk for screenshots / GitHub
    out_dir = Path("runs") / run_id_bm25
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    (out_dir / "analysis_url.txt").write_text(f"/analysis/{run_id_bm25}", encoding="utf-8")

    return HTMLResponse(content=html)


@app.get("/analysis/{run_id}", response_class=HTMLResponse)
def analysis(run_id: str, request: Request, q: int = 0):
    """
    Visual failure analysis:
    - click a query
    - see retrieved chunks + scores
    - see whether relevant chunk was hit and at what rank
    - heuristic "why" explanation
    """
    from app.core.ingest import ingest_folder
    from app.core.benchmarks import build_benchmark_from_docs
    from app.core.retrieval import BM25Retriever

    chunks = ingest_folder(folder="data/docs", chunk_size=120, overlap=25)
    chunk_map = {c.chunk_id: c.text for c in chunks}


        # Doc labels for enterprise-style debugging
    DOC_LABELS = {
        "api_style_legacy": "LEGACY/OUTDATED",
        "api_errors_legacy": "LEGACY/OUTDATED",
        "legacy_logging_guidelines": "LEGACY/OUTDATED",
        "logging_minimal_legacy": "LEGACY/OUTDATED",
        "incident_calendar_policy": "POLICY (NOT INCIDENT STEPS)",
        "operations_policy": "POLICY (NOT INCIDENT STEPS)",
        "logging_data_platform": "BATCH LOGGING (NOT API TRACING)",
        "security_logging_exceptions": "EXCEPTION POLICY",
    }

    retriever = BM25Retriever(chunk_map)
    bench = build_benchmark_from_docs(chunks)

    if not bench:
        return HTMLResponse("<h3>No benchmark queries found.</h3>")

    # clamp selection index
    selected_idx = max(0, min(int(q), len(bench) - 1))
    selected = bench[selected_idx]

    k_rank = 10
    results = retriever.search(selected.query, k=k_rank)

    relevant = set(selected.relevant_chunk_ids)
    relevant_ids = ", ".join(sorted(list(relevant))) if relevant else "(none)"
    relevant_count = len(relevant)

    # Build retrieved table rows
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
            "rank": i,
            "chunk_id": r.doc_id,
            "score": r.score,
            "preview": preview,
            "is_relevant": is_rel,
            "label": label
        })


    # Simple “why” heuristic for misses: term overlap between query and relevant chunk
    why = ""
    if not hit and relevant:
        # take first relevant chunk
        rel_id = list(relevant)[0]
        rel_text = chunk_map.get(rel_id, "")
        q_terms = set([t.lower().strip(".,!?;:()[]{}\"'") for t in selected.query.split() if t.strip()])
        d_terms = set([t.lower().strip(".,!?;:()[]{}\"'") for t in rel_text.split() if t.strip()])
        overlap = sorted(list(q_terms.intersection(d_terms)))
        if not overlap:
            why = "Low lexical overlap: query terms do not appear in the relevant chunk (BM25 relies on term matches). Try synonyms, chunking tweaks, or hybrid retrieval."
        else:
            why = f"Some overlap exists ({', '.join(overlap)}), but other chunks scored higher. Try smaller chunks or add more distinctive terms."

    return templates.TemplateResponse(
        "analysis.html",
        {
            "request": request,
            "run_id": run_id,
            "queries": bench,
            "selected_idx": selected_idx,
            "selected": selected,
            "k_rank": k_rank,
            "relevant_count": relevant_count,
            "relevant_ids": relevant_ids,
            "retrieved": rows,
            "hit": hit,
            "hit_rank": hit_rank,
            "why": why
        },
    )


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    """
    Leaderboard of all runs + key metrics.
    """
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
                        # pull retriever name from config_json
            cfg_row = conn.execute(
                "SELECT config_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            retriever_name = "unknown"
            if cfg_row and cfg_row["config_json"]:
                try:
                    cfg = json.loads(cfg_row["config_json"])
                    retriever_name = cfg.get("retriever", "unknown")
                except Exception:
                    retriever_name = "unknown"

            out.append({
                "run_id": run_id,
                "created_at": row["created_at"],
                "retriever": retriever_name,
                "metrics": [{"name": m["metric_name"], "value": round(float(m["metric_value"]), 4)} for m in metrics],
            })


    finally:
        conn.close()

    return templates.TemplateResponse("runs.html", {"request": request, "runs": out})


@app.get("/open-report/{run_id}", response_class=HTMLResponse)
def open_report(run_id: str):
    """
    Opens the saved HTML report generated by /demo-run.
    """
    report_path = Path("runs") / run_id / "report.html"
    if not report_path.exists():
        return HTMLResponse(f"<h3>Report not found for {run_id}</h3>", status_code=404)
    return HTMLResponse(report_path.read_text(encoding="utf-8"))

@app.get("/compare", response_class=HTMLResponse)
def compare():
    """
    Simple comparison chart (MRR@10) across recent runs by retriever.
    """
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
            run_id = row["run_id"]
            created_at = row["created_at"]

            retriever = "unknown"
            if row["config_json"]:
                try:
                    cfg = json.loads(row["config_json"])
                    retriever = cfg.get("retriever", "unknown")
                except Exception:
                    retriever = "unknown"

            mrr = conn.execute(
                "SELECT metric_value FROM metrics WHERE run_id = ? AND metric_name = 'MRR@10'",
                (run_id,),
            ).fetchone()
            if mrr:
                points.append((created_at, retriever, float(mrr["metric_value"]), run_id))

    finally:
        conn.close()

    # Split by retriever
    by_ret = {}
    for created_at, retriever, mrr, run_id in points:
        by_ret.setdefault(retriever, []).append((created_at, mrr, run_id))

    fig = go.Figure()
    for retriever, xs in by_ret.items():
        xs_sorted = sorted(xs, key=lambda x: x[0])
        fig.add_trace(go.Scatter(
            x=[a for a,_,__ in xs_sorted],
            y=[b for _,b,__ in xs_sorted],
            mode="lines+markers",
            name=retriever
        ))

    fig.update_layout(
        title="MRR@10 by Retriever (Recent Runs)",
        xaxis_title="created_at (UTC)",
        yaxis_title="MRR@10",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return HTMLResponse(f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>RAGBench Compare</title>
        <style>
          body {{ font-family: system-ui; background:#0b0f19; color:#e6e6e6; padding:24px; }}
          .wrap {{ max-width:1200px; margin:0 auto; }}
          .card {{ background:#111827; border:1px solid #1f2937; border-radius:18px; padding:16px; }}
          a {{ color:#93c5fd; text-decoration:none; }}
          a:hover {{ text-decoration:underline; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h2 style="margin:0;">RAGBench Comparison</h2>
            <div><a href="/runs">Back to /runs</a></div>
          </div>
          <div class="card">{chart}</div>
        </div>
      </body>
    </html>
    """)
@app.get("/regression", response_class=HTMLResponse)
def regression():
    conn = get_conn()
    try:
        result = detect_regression(conn)
    finally:
        conn.close()

    if result["status"] == "insufficient_data":
        msg = "Not enough runs to detect regression yet."
    elif result["status"] == "regression":
        msg = f"""
        🚨 REGRESSION DETECTED<br><br>
        Latest run <b>{result['latest_run']}</b> has MRR@10 = <b>{result['latest_value']}</b><br>
        Best recent run <b>{result['best_run']}</b> had MRR@10 = <b>{result['best_value']}</b><br><br>
        Recommendation: investigate retriever/config changes before shipping.
        """
    else:
        msg = f"""
        ✅ No regression detected.<br><br>
        Latest run MRR@10 = <b>{result['latest_value']}</b><br>
        Best recent MRR@10 = <b>{result['best_value']}</b>
        """

    return HTMLResponse(f"""
    <html>
      <body style="font-family: system-ui; background:#0b0f19; color:#e6e6e6; padding:24px;">
        <h2>RAGBench — Regression Guard</h2>
        <div style="margin-top:16px; font-size:16px;">{msg}</div>
        <div style="margin-top:20px;">
          <a href="/runs" style="color:#93c5fd;">Back to Runs</a>
        </div>
      </body>
    </html>
    """)
