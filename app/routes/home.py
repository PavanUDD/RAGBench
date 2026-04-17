from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def home():
    p = Path("app/templates/home.html")
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h2>RAGBench ✅</h2>")
