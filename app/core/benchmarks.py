from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict

from app.core.ingest import Chunk


@dataclass
class BenchmarkQuery:
    query: str
    relevant_chunk_ids: Set[str]


def _chunks_for_doc(by_doc: Dict[str, List[Chunk]], doc_id: str) -> Set[str]:
    return set([c.chunk_id for c in by_doc.get(doc_id, [])])


def build_benchmark_from_docs(chunks: List[Chunk]) -> List[BenchmarkQuery]:
    """
    Internal Docs QA benchmark: realistic employee questions + gold relevant chunks.
    This is what makes the project "real": we measure retrieval quality against internal documentation.
    """
    by_doc: Dict[str, List[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    # Gold sets (relevant sources)
    onboarding = _chunks_for_doc(by_doc, "internal_onboarding")
    incident = _chunks_for_doc(by_doc, "incident_response")
    api = _chunks_for_doc(by_doc, "api_standards")
    obs = _chunks_for_doc(by_doc, "logging_observability")
    rag_playbook = _chunks_for_doc(by_doc, "rag_assistant_playbook")
    security = _chunks_for_doc(by_doc, "security_basics")

    # Keep your original AWS-ish docs too (if present)
    aws_rag = _chunks_for_doc(by_doc, "aws_rag_basics")
    aws_obs = _chunks_for_doc(by_doc, "aws_observability_genai")

    queries: List[BenchmarkQuery] = [
        # Onboarding
        BenchmarkQuery("Where are services located in the repo and what is the local setup?", onboarding),
        BenchmarkQuery("What environments do we have and what should never be done in prod?", onboarding),

        # Incident response
        BenchmarkQuery("What is SEV-1 and what should happen in the first 15 minutes?", incident),
        BenchmarkQuery("What do we write after an incident and what follow-ups are required?", incident),

        # API standards
        BenchmarkQuery("What is our standard error response format?", api),
        BenchmarkQuery("When should we version an API and what should we log for each request?", api),

        # Logging & observability
        BenchmarkQuery("What fields must be included in structured logs for tracing?", obs),
        BenchmarkQuery("For RAG systems, what retrieval details should we log?", obs),

        # RAG playbook
        BenchmarkQuery("What is the common failure mode in RAG assistants and how do we mitigate it?", rag_playbook),
        BenchmarkQuery("Why should we track Recall@k, MRR, and nDCG for retrieval?", rag_playbook),

        # Security
        BenchmarkQuery("What should we never log and what principle should access follow?", security),
        BenchmarkQuery("How do security incidents relate to the SEV process and documentation?", security),

        # Optional: include AWS RAG docs if present
        BenchmarkQuery("Explain what RAG is and why it improves factual accuracy.", aws_rag or rag_playbook),
        BenchmarkQuery("What does observability mean for GenAI systems?", aws_obs or obs),

                # Paraphrases / realistic variations (harder)
        BenchmarkQuery("If prod is risky, what environment should we test destructive changes in?", onboarding),
        BenchmarkQuery("During a major incident, who leads and what are the first actions?", incident),
        BenchmarkQuery("What should every API log include for tracing and debugging?", api),
        BenchmarkQuery("What is the current standard error shape returned by our APIs?", api),
        BenchmarkQuery("Which IDs must be present for distributed tracing across services?", obs),
        BenchmarkQuery("For RAG debugging, what should we capture about retrieval results?", obs),
        BenchmarkQuery("What is least privilege and where should secrets be stored?", security),
        BenchmarkQuery("What do we require after an incident to prevent repeat failures?", incident),





    ]

    # Filter out any queries whose gold set is empty (if a doc file is missing)
    queries = [q for q in queries if q.relevant_chunk_ids]
    return queries
