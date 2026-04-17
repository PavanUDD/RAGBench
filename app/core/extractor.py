from __future__ import annotations
from pathlib import Path


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    try:
        import io
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {e}")


def extract_text_from_txt(content: bytes) -> str:
    """Decode plain text."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return content.decode(enc)
        except Exception:
            continue
    raise ValueError("Could not decode text file")


def extract_text(filename: str, content: bytes) -> str:
    """Route to correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(content)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(content)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")
