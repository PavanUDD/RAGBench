from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def _read_text_files(folder: str) -> List[tuple[str, str]]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Docs folder not found: {p.resolve()}")


    files_txt = [f for f in p.glob("*.txt") if f.is_file()]
    files_md = [f for f in p.glob("*.md") if f.is_file()]
    files = sorted(files_txt + files_md)
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {p.resolve()}")

    out: List[tuple[str, str]] = []
    for f in files:
        doc_id = f.stem
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        out.append((doc_id, text))
    return out


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 40) -> List[str]:
    """
    Simple word-based chunking (fast + local).
    chunk_size and overlap are in WORDS.
    """
    words = [w for w in text.split() if w.strip()]
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks


def ingest_folder(folder: str = "data/docs", chunk_size: int = 250, overlap: int = 40) -> List[Chunk]:
    docs = _read_text_files(folder)
    chunks: List[Chunk] = []
    for doc_id, text in docs:
        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunk_id = f"{doc_id}::c{i:03d}"
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=part))
    return chunks
