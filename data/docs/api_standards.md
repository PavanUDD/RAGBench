# API Standards (Internal)

## Naming and structure
- Use REST with clear resource naming (/users, /orders).
- Avoid verbs in endpoints when possible.

## Tracing and auditability
- Include request_id for every request and propagate it across services.
- For sensitive operations, record audit trail fields (actor, action, resource, timestamp).

## Errors (current standard)
Return consistent error format:
{
  code: string,
  message: string,
  details: object (optional)
}

## Versioning
- Version APIs when breaking changes are introduced.
- Keep backward compatibility when possible.

## Logging (minimum)
- Log request_id, route, status_code, and latency_ms.
- Do NOT log secrets or tokens.
