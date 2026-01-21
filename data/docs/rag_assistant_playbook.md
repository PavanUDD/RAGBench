# Internal Docs Assistant (RAG) Playbook

## What breaks in production
Common failure mode: retrieval returns wrong chunks, causing confident wrong answers.

Why it happens:
- chunks are too big or too small
- ambiguous queries map to multiple policies
- legacy docs contain similar keywords
- missing request_id/audit fields reduces debugging ability

## How we mitigate
- Improve chunking and add overlap.
- Add hard negatives to evaluation (confuser docs).
- Track retrieval quality over time:
  - coverage (did we retrieve the right sources?)
  - ranking quality (did the right sources appear early?)
- Provide an audit trail:
  query -> retrieved chunks -> citations -> config version.

## Release discipline
Do not ship changes to retrievers or chunking without evaluation runs.
If metrics regress, rollback or fix before shipping.
