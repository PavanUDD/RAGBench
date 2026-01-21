from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float


class TfidfRetriever:
    def __init__(self, docs: Dict[str, str]):
        self.doc_ids = list(docs.keys())
        self.texts = [docs[i] for i in self.doc_ids]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 10) -> List[RetrievedDoc]:
        q_vec = self.vectorizer.transform([query])
        # cosine similarity since TF-IDF vectors are normalized-ish
        scores = (self.doc_matrix @ q_vec.T).toarray().reshape(-1)
        idx = np.argsort(scores)[::-1][:k]
        return [RetrievedDoc(doc_id=self.doc_ids[i], score=float(scores[i])) for i in idx]
