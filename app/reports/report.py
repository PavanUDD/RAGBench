from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class MetricPoint:
    name: str
    value: float


def build_dashboard_html(run_id: str, title: str, metrics: List[MetricPoint]) -> str:
    # Simple “pro” look: dark template
    pio.templates.default = "plotly_dark"

    labels = [m.name for m in metrics]
    values = [m.value for m in metrics]

    fig = go.Figure(
        data=[go.Bar(x=labels, y=values)]
    )
    fig.update_layout(
        title=f"{title} — {run_id}",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Self-contained page (except plotlyjs from CDN; no cost, just a free JS file)
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAGBench Report - {run_id}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: #0b0f19;
      color: #e6e6e6;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 18px;
      padding: 18px 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 6px 0;
    }}
    .sub {{
      color: #a7b0c0;
      font-size: 13px;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.35);
      color: #c7d2fe;
      font-size: 12px;
      margin-left: 8px;
      vertical-align: middle;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>RAGBench Report <span class="pill">Local • $0</span></h1>
      <div class="sub">Run: <b>{run_id}</b> • Generated: {generated}</div>
      <div class="grid">
        {chart_html}
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html
