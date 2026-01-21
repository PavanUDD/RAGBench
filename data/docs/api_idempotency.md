# API Idempotency (Internal)

- For create/charge actions, support idempotency keys.
- Retries should not cause duplicates.
- Log idempotency key with request_id for traceability.
