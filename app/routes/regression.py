from __future__ import annotations
import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.db.database import get_conn

router = APIRouter()

def _detect_regression(conn, metric="MRR@10", min_history=3, tolerance=0.02, retriever="HYBRID"):
    rows = conn.execute("""SELECT r.run_id,r.created_at,r.config_json,m.metric_value
        FROM runs r JOIN metrics m ON r.run_id=m.run_id
        WHERE m.metric_name=? ORDER BY r.created_at DESC LIMIT 50""", (metric,)).fetchall()

    filtered = []
    for row in rows:
        try:
            if json.loads(row["config_json"] or "{}").get("retriever") == retriever:
                filtered.append(row)
        except: pass

    if not filtered: return {"status":"insufficient_data"}
    try:
        sig = json.loads(filtered[0]["config_json"] or "{}").get("benchmark_signature")
        if sig:
            filtered = [r for r in filtered if json.loads(r["config_json"] or "{}").get("benchmark_signature") == sig]
    except: pass

    if len(filtered) < min_history: return {"status":"insufficient_data","retriever":retriever}
    latest = filtered[0]
    best   = max(filtered[1:], key=lambda x: float(x["metric_value"]))
    lv, bv = float(latest["metric_value"]), float(best["metric_value"])
    if lv < bv - tolerance:
        return {"status":"regression","latest_run":latest["run_id"],"latest_value":lv,"best_run":best["run_id"],"best_value":bv}
    return {"status":"ok","latest_run":latest["run_id"],"latest_value":lv,"best_value":bv}


def _status_html(result):
    if result["status"] == "insufficient_data":
        return """<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:24px">⏳</span><span style="font-size:18px;font-weight:700;color:#f59e0b">Collecting Data</span></div>
          <p style="color:#8b949e">Run at least 3 benchmarks to enable regression detection.</p>"""
    elif result["status"] == "regression":
        return f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:24px">🚨</span><span style="font-size:18px;font-weight:700;color:#ef4444">REGRESSION DETECTED</span></div>
          <p style="color:#8b949e;margin-bottom:16px">Latest HYBRID run shows degraded quality.</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:400px">
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px">
              <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Latest MRR@10</div>
              <div style="font-size:22px;font-weight:800;color:#ef4444">{result['latest_value']:.4f}</div></div>
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px">
              <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Best MRR@10</div>
              <div style="font-size:22px;font-weight:800;color:#10b981">{result['best_value']:.4f}</div></div></div>"""
    else:
        return f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <span style="font-size:24px">✅</span><span style="font-size:18px;font-weight:700;color:#10b981">No Regression Detected</span></div>
          <p style="color:#8b949e;margin-bottom:16px">HYBRID retrieval quality is stable.</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:400px">
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px">
              <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Latest MRR@10</div>
              <div style="font-size:22px;font-weight:800;color:#a78bfa">{result['latest_value']:.4f}</div></div>
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px">
              <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Best MRR@10</div>
              <div style="font-size:22px;font-weight:800;color:#10b981">{result['best_value']:.4f}</div></div></div>"""


@router.get("/regression", response_class=HTMLResponse)
def regression():
    conn = get_conn()
    try: result = _detect_regression(conn)
    finally: conn.close()

    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><title>RAGBench — Regression Guard</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:#060910;color:#e6edf3}}
nav{{display:flex;align-items:center;justify-content:space-between;padding:16px 40px;border-bottom:1px solid #21262d;background:rgba(6,9,16,0.9)}}
.logo{{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;text-decoration:none;color:#e6edf3}}
.logo-icon{{width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:8px;display:flex;align-items:center;justify-content:center}}
.nav-links a{{padding:7px 14px;border-radius:8px;font-size:14px;color:#8b949e;text-decoration:none;margin-left:4px}}.nav-links a:hover{{background:#161b22;color:#e6edf3}}
.cta{{padding:8px 18px;background:#3b82f6;border-radius:8px;font-size:14px;font-weight:600;color:white;text-decoration:none}}
.page{{max-width:900px;margin:0 auto;padding:40px}}h1{{font-size:24px;font-weight:800;margin-bottom:6px}}
.sub{{font-size:14px;color:#8b949e;margin-bottom:32px}}.card{{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:28px}}
.back{{display:inline-flex;align-items:center;gap:6px;margin-top:24px;color:#3b82f6;text-decoration:none;font-size:14px;font-weight:500}}</style>
</head><body>
<nav><a href="/" class="logo"><div class="logo-icon">⚡</div>&nbsp;RAGBench</a>
<div class="nav-links"><a href="/dashboard">Dashboard</a><a href="/runs">Runs</a><a href="/compare">Compare</a><a href="/regression">Regression Guard</a></div>
<a href="/demo-run" class="cta">▶ New Run</a></nav>
<div class="page"><h1>Regression Guard</h1>
<div class="sub">Monitors HYBRID MRR@10 — alerts when quality drops below best recent run</div>
<div class="card">{_status_html(result)}<a href="/runs" class="back">← Back to Runs</a></div>
</div></body></html>""")
