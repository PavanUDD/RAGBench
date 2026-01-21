import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class Document:
    id: str
    text: str


@dataclass
class QueryItem:
    query: str
    relevant_docs: List[str]


@dataclass
class Dataset:
    documents: List[Document]
    queries: List[QueryItem]


def load_dataset(path: str = "data/dataset.json") -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p.resolve()}")

    raw: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    docs = [Document(id=d["id"], text=d["text"]) for d in raw["documents"]]
    queries = [QueryItem(query=q["query"], relevant_docs=q["relevant_docs"]) for q in raw["queries"]]
    return Dataset(documents=docs, queries=queries)
