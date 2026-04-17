from __future__ import annotations
import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.db.database import get_conn

router = APIRouter()

@router.get("/compare", response_class=HTMLResponse)
def compare():
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.templates.default = "plotly_dark"

    conn = get_conn()
    try:
        rows = conn.execute("SELECT run_id, created_at, config_json FROM runs ORDER BY created_at DESC LIMIT 30").fetchall()
        points = []
        for row in rows:
            retriever = "unknown"
            if row["config_json"]:
                try: retriever = json.loads(row["config_json"]).get("retriever","unknown")
                except: pass
            mrr = conn.execute("SELECT metric_value FROM metrics WHERE run_id=? AND metric_name='MRR@10'", (row["run_id"],)).fetchone()
            if mrr:
                points.append((row["created_at"], retriever, float(mrr["metric_value"])))
    finally:
        conn.close()

    by_ret: dict = {}
    for ts, ret, val in points:
        by_ret.setdefault(ret, []).append((ts, val))

    COLORS = {"HYBRID":"#a78bfa","TFIDF":"#60a5fa","BM25":"#34d399"}
    fig = go.Figure()
    for ret, xs in by_ret.items():
        xs = sorted(xs)
        fig.add_trace(go.Scatter(x=[a for a,_ in xs], y=[b for _,b in xs],
            mode="lines+markers", name=ret, line=dict(color=COLORS.get(ret,"#fff"),width=2), marker=dict(size=8)))
    fig.update_layout(title="MRR@10 by Retriever (Recent Runs)",
        xaxis_title="Timestamp (UTC)", yaxis_title="MRR@10",
        height=520, margin=dict(l=40,r=40,t=60,b=40), legend=dict(bgcolor="rgba(0,0,0,0)"))
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench Compare</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3}}
nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
.logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
.logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
.nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}.nav-links a:hover{{background:#161b22;color:#e6edf3}}
.cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
.page{{max-width:1200px;margin:0 auto;padding:40px}}h1{{font-size:24px;font-weight:800;margin-bottom:6px}}
.sub{{font-size:14px;color:#8b949e;margin-bottom:28px}}.card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:20px}}</style>
</head><body>
<nav><a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
<div class="nav-links"><a href="/dashboard">Dashboard</a><a href="/runs">Runs</a><a href="/compare">Compare</a><a href="/regression">Regression Guard</a></div>
<a href="/demo-run" class="cta">▶ New Run</a></nav>
<div class="page"><h1>Retriever Comparison</h1>
<div class="sub">MRR@10 trend · BM25 vs TF-IDF vs Hybrid Dense+Sparse</div>
<div class="card">{chart}</div></div></body></html>""")
