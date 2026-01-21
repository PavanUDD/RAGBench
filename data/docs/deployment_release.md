# Deployment & Release (Internal)

- Environments: dev -> staging -> prod.
- Use feature flags for risky changes.
- Rollbacks must be documented with reason and timestamp.
- All releases require passing unit tests and at least one integration check.
- Avoid deploying during active incidents.
