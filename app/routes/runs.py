from __future__ import annotations
import json
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.db.database import get_conn

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    conn = get_conn()
    try:
        rows = conn.execute("SELECT run_id, created_at FROM runs ORDER BY created_at DESC LIMIT 50").fetchall()
        out = []
        for row in rows:
            rid = row["run_id"]
            metrics = conn.execute("SELECT metric_name, metric_value FROM metrics WHERE run_id=? ORDER BY metric_name", (rid,)).fetchall()
            cfg_row = conn.execute("SELECT config_json FROM runs WHERE run_id=?", (rid,)).fetchone()
            retriever = "unknown"
            if cfg_row and cfg_row["config_json"]:
                try:
                    retriever = json.loads(cfg_row["config_json"]).get("retriever", "unknown")
                except Exception:
                    pass
            out.append({"run_id":rid,"created_at":row["created_at"],"retriever":retriever,
                        "metrics":[{"name":str(m["metric_name"]),"value":round(float(m["metric_value"]),4)} for m in metrics]})
    finally:
        conn.close()
    return templates.TemplateResponse("runs.html", {"request":request,"runs":out})
