# Engineering Onboarding (Internal)

## Repo layout
- mono-repo contains services under /services.
- shared libraries under /libs.
- docs under /docs.

## Local setup
1) Create Python venv
2) Install requirements
3) Run unit tests before pushing
4) Use dev environment defaults

## Environments
- dev: for local testing
- staging: pre-prod validation
- prod: customer-facing
Never test destructive changes in prod.

## Secrets
- never commit secrets
- use environment variables locally
- use a secrets manager in production
