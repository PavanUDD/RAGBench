\# RAGBench



RAGBench is a local, zero-cost evaluation and observability toolkit for Retrieval-Augmented Generation (RAG) systems.



It is designed to help teams \*\*measure retrieval quality\*\*, \*\*debug failure modes\*\*, and \*\*prevent regressions\*\* before deploying GenAI assistants in production.



---



\## Why RAGBench?



In real-world GenAI systems, most failures are not caused by the language model — they are caused by \*\*retrieval returning the wrong document\*\*, often a legacy or non-authoritative one.



RAGBench solves this by:

\- benchmarking retrievers with ground-truth queries

\- exposing failure cases visually

\- tracking metrics over time

\- detecting regressions before shipping



---



\## High-Level Architecture



!\[RAGBench Architecture](assets/ragbench\_architecture.png)



RAGBench ingests internal documentation, chunks it into retrievable units, evaluates multiple retrieval strategies (BM25 and TF-IDF) against a benchmark query set, and tracks retrieval quality metrics over time.



Each evaluation run is stored with its configuration and metrics, enabling comparison, failure analysis, and regression detection.



---



\## Key Features



\- \*\*Retriever Benchmarking\*\*

&nbsp; - Compare BM25 and TF-IDF on realistic internal queries

&nbsp; - Metrics: Recall@k, MRR@k, nDCG@k



\- \*\*Failure Analysis\*\*

&nbsp; - Inspect retrieved chunks per query

&nbsp; - Identify legacy / outdated documents surfaced by retrievers



\- \*\*Experiment Tracking\*\*

&nbsp; - Each run stored with configuration and metrics in SQLite

&nbsp; - Reproducible and comparable runs



\- \*\*Regression Guard\*\*

&nbsp; - Detects performance drops using apples-to-apples benchmark signatures

&nbsp; - Prevents shipping degraded retrieval configurations



\- \*\*Interactive Dashboards\*\*

&nbsp; - Runs leaderboard

&nbsp; - Retriever comparison trends

&nbsp; - Failure analysis drill-downs



---



\## Tech Stack



\- Python

\- FastAPI

\- SQLite

\- BM25 and TF-IDF retrieval

\- Plotly (visualizations)

\- Jinja2 (HTML reports)



---



\## Running Locally



```bash

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn app.main:app --reload



