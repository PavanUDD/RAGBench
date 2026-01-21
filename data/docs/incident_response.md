# Incident Response (SEV Guide)

## Severity levels
- SEV-1: customer impact and/or security risk. Immediate response required.
- SEV-2: partial impact or degraded service requiring same-day mitigation.
- SEV-3: limited impact, workaround exists, schedule fix.

## First 15 minutes checklist
1) Assign incident lead and communications lead.
2) Define customer impact and affected systems.
3) Collect key signals: error rate, latency, saturation, and recent deploys.
4) If a recent deploy is suspected, rollback first, then investigate root cause.
5) Start a timeline and record every action (who/what/when).

## Communication
- Post updates every 15–30 minutes for SEV-1.
- Communicate mitigation and next checkpoint.

## Post-incident requirements
- Blameless postmortem within 48–72 hours.
- Action items must include: prevention (tests/guardrails) + detection (alerts/metrics).
- Add regression tests for the incident-causing bug.
