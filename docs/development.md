# Development guide

This guide covers local installation, configuration, and verification. For system behavior, see the [technical reference](technical-reference.md).

## Prerequisites

- Python 3.10 or newer
- Docker for isolated local workspace sessions
- Access to the LeanWorks GCP project and development credentials
- PostgreSQL or Cloud SQL access for database-backed operations
- Access to leanworks-hub for document and platform workflows

Cloud SQL Auth Proxy is needed for database-backed local work. Application startup attempts to launch it when it is installed and port 5432 is not already in use.

## Install

```bash
git clone https://github.com/leanworksai/leanworks.git
cd leanworks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e . --no-deps
```

## Configure local services

Place the development service-account file at `gcp_credential_dev.json` in the repository root. Do not commit credential files.

Set the local environment:

```bash
export ENVIRONMENT=local
export DB_HOST=127.0.0.1
```

By default, local development expects leanworks-hub at `http://localhost:3001`. Point to a different environment only when needed:

```bash
export LEANWORKS_HUB_URL=https://dev.leanworks.ai
```

Relevant service configuration includes:

```bash
GCP_PROJECT_ID=your-project
GCP_VECTOR_SEARCH_LOCATION=us-central1
GCP_VECTOR_SEARCH_COLLECTION_TEXT=leanworks-multimodal
GCP_VECTOR_SEARCH_COLLECTION_CODES=leanworks-codes
GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES=leanworks-tool-responses
FIRESTORE_DATABASE_NAME=your-firestore-database
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_credential.json
LEANWORKS_HUB_URL=http://localhost:3001
LEANWORKS_API_KEY=your-api-key
```

`get_hub_url()` chooses an environment-aware default when `LEANWORKS_HUB_URL` is not set: localhost for local development, the development site in dev, the production site in production, and the hub service address inside Kubernetes.

GCP Vector Search is the configured retrieval backend. There is no runtime feature flag selecting an alternative backend.

## Run the API

```bash
python run.py
```

The local server listens on `http://localhost:8082`.

## Run tests

Run the project test suite through the managed script:

```bash
./scripts/run_tests.sh
```

Run only the documentation contract checks:

```bash
python -m pytest tests/test_readme_drift.py
python scripts/check_readme_drift.py
```

## Troubleshooting

- **Credentials are not found:** confirm that `gcp_credential_dev.json` exists at the repository root or that `GOOGLE_APPLICATION_CREDENTIALS` points to the intended file.
- **Database connections fail:** confirm that Cloud SQL Auth Proxy is listening on port 5432 and that the database variables are correct.
- **Hub calls fail:** run leanworks-hub locally or set `LEANWORKS_HUB_URL` to an accessible environment.
- **Workspace sessions fail:** confirm Docker is running and that the local user can build and start containers.

## Deployment

Local development and production deployment use different credential paths and service addresses. Follow the [deployment guide](../deploy/README.md) for container builds, Kubernetes resources, and environment-specific behavior.
