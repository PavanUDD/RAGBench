# Logging & Observability (Internal)

Observability includes metrics, logs, and traces.

## Metrics (golden signals)
- Latency, error rate, saturation, traffic.
- Track SLOs and error budgets.

## Logs (structured)
- Use structured JSON logs.
- Always include: request_id, component, status_code, latency_ms.
- Include user_id only if allowed and redacted appropriately.

## Traces
- Propagate trace_id across service boundaries.
- Spans should include external dependencies (DB, cache, downstream APIs).

## RAG-specific observability
For RAG systems, log:
- user_query
- retrieved chunk_ids and similarity scores
- retrieval config (chunk_size, overlap, retriever type)
- prompt version (if generating answers)
- latency for retrieval and generation
This is required for debugging hallucinations and for audits.
