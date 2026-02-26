# aumos-mlops-lifecycle

[![CI](https://github.com/aumos-enterprise/aumos-mlops-lifecycle/actions/workflows/ci.yml/badge.svg)](https://github.com/aumos-enterprise/aumos-mlops-lifecycle/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aumos-enterprise/aumos-mlops-lifecycle/branch/main/graph/badge.svg)](https://codecov.io/gh/aumos-enterprise/aumos-mlops-lifecycle)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> End-to-end MLOps/LLMOps lifecycle management — experiment tracking, zero-downtime model deployment, feature store integration, and automated retraining for AumOS Enterprise.

## Overview

`aumos-mlops-lifecycle` is the central orchestration service for the AumOS MLOps/LLMOps
platform. It manages the complete lifecycle of machine learning models from experimentation
through production deployment and ongoing retraining. Every ML model running in production
on the AumOS platform passes through this service's deployment and monitoring pipelines.

The service integrates MLflow for experiment tracking, providing tenant-isolated experiment
namespaces where data science teams can log runs, compare metrics, and promote the best
model version to deployment. Deployments support four strategies — canary, A/B testing,
shadow, and blue-green — all with automated rollback capabilities and real-time health
monitoring.

Feature engineering is managed through Feast integration, allowing teams to define,
version, and materialize feature sets from batch and streaming sources. Automated model
retraining is triggered by drift signals from `aumos-drift-detector`, cron schedules, or
manual operator requests, ensuring production models stay accurate as data distributions
evolve over time.

**Product:** MLOps and LLMOps Platform (Product 7)
**Tier:** Tier 3 — Intelligence Operations
**Phase:** 3 (Months 12-18)

## Architecture

```
aumos-common ─────────────────────────────────────────────────────────►
aumos-proto  ─────────────────────────────────────────────────────────►
aumos-model-registry ─────────────────────────────────────────────────► aumos-mlops-lifecycle
                                                                                │
                                        ┌───────────────────────────────────────┤
                                        ▼               ▼               ▼       ▼
                               aumos-drift-detector  aumos-observability  aumos-llm-serving
                               aumos-event-bus       aumos-agent-framework
```

This service follows AumOS hexagonal architecture:

- `api/` — FastAPI routes (thin layer, delegates all logic to services)
- `core/` — Business logic with no framework dependencies (services, ORM models, interfaces)
- `adapters/` — External integrations (PostgreSQL via SQLAlchemy, Kafka, MLflow, Feast)

### Key Capabilities

| Capability | Implementation |
|------------|----------------|
| Experiment tracking | MLflow with per-tenant namespace isolation |
| Model deployment | Canary, A/B, shadow, blue-green strategies |
| Feature store | Feast for batch and streaming feature materialization |
| Automated retraining | Drift-triggered, scheduled, and manual retraining jobs |
| Tenant isolation | PostgreSQL RLS + MLflow experiment namespacing |
| Event streaming | Kafka events for all lifecycle state changes |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to AumOS internal PyPI for `aumos-common` and `aumos-proto`

### Local Development

```bash
# Clone the repo
git clone https://github.com/aumos-enterprise/aumos-mlops-lifecycle.git
cd aumos-mlops-lifecycle

# Set up environment
cp .env.example .env
# Edit .env with your local values

# Install dependencies
make install

# Start infrastructure (PostgreSQL, Redis, Kafka, MLflow)
docker compose -f docker-compose.dev.yml up -d

# Run the service
uvicorn aumos_mlops_lifecycle.main:app --reload
```

The service will be available at `http://localhost:8000`.

Health check: `http://localhost:8000/live`
Readiness probe: `http://localhost:8000/ready`
API docs: `http://localhost:8000/docs`
MLflow UI: `http://localhost:5000`

## API Reference

### Authentication

All endpoints require a Bearer JWT token and tenant header:

```
Authorization: Bearer <token>
X-Tenant-ID: <tenant-uuid>
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/live` | Liveness probe |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/experiments` | Create a new experiment |
| GET | `/api/v1/experiments` | List experiments for tenant |
| GET | `/api/v1/experiments/{id}` | Get experiment by ID |
| POST | `/api/v1/experiments/{id}/runs` | Log a run to an experiment |
| GET | `/api/v1/experiments/{id}/runs` | List runs for an experiment |
| POST | `/api/v1/deployments` | Create a model deployment |
| GET | `/api/v1/deployments` | List deployments for tenant |
| GET | `/api/v1/deployments/{id}` | Get deployment status |
| POST | `/api/v1/deployments/{id}/rollback` | Roll back a deployment |
| POST | `/api/v1/feature-sets` | Create a feature set |
| GET | `/api/v1/feature-sets` | List feature sets |
| GET | `/api/v1/feature-sets/{id}` | Get feature set |
| POST | `/api/v1/retraining-jobs` | Trigger a retraining job |
| GET | `/api/v1/retraining-jobs` | List retraining jobs |
| GET | `/api/v1/retraining-jobs/{id}` | Get retraining job status |

Full OpenAPI spec available at `/docs` when running locally.

### Example: Create Experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "churn-prediction-v3",
    "description": "XGBoost churn model with RFM features",
    "tags": {"team": "data-science", "project": "churn"}
  }'
```

### Example: Deploy a Model

```bash
curl -X POST http://localhost:8000/api/v1/deployments \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_version": "3",
    "strategy": "canary",
    "target_environment": "production",
    "traffic_split": {"stable": 90, "canary": 10},
    "health_check_url": "https://models.internal/churn/health"
  }'
```

### Example: Trigger Retraining

```bash
curl -X POST http://localhost:8000/api/v1/retraining-jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "550e8400-e29b-41d4-a716-446655440000",
    "trigger_type": "manual"
  }'
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_SERVICE_NAME` | `aumos-mlops-lifecycle` | Service identifier |
| `AUMOS_ENVIRONMENT` | `development` | Runtime environment |
| `AUMOS_DATABASE__URL` | — | PostgreSQL connection string |
| `AUMOS_KAFKA__BROKERS` | `localhost:9092` | Kafka broker list |
| `AUMOS_MLOPS_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL |
| `AUMOS_MLOPS_FEAST_REGISTRY_PATH` | `data/registry.db` | Feast registry path or GCS URI |
| `AUMOS_MLOPS_CANARY_ERROR_THRESHOLD` | `0.05` | Error rate that triggers auto-rollback |
| `AUMOS_MLOPS_CANARY_STEP_PERCENT` | `10` | Traffic increment per canary progression step |
| `AUMOS_MLOPS_MAX_CONCURRENT_RETRAINING_JOBS` | `5` | Max parallel retraining jobs per tenant |

See `src/aumos_mlops_lifecycle/settings.py` for the full settings class.

## Development

### Running Tests

```bash
# Full test suite with coverage
make test

# Fast run (stop on first failure)
make test-quick

# Run with HTML coverage report
pytest tests/ -v --cov --cov-report=html
open htmlcov/index.html
```

### Linting and Formatting

```bash
# Check for issues
make lint

# Auto-fix formatting
make format

# Type checking
make typecheck
```

### Adding Dependencies

```bash
# Add a runtime dependency
# Edit pyproject.toml -> [project] dependencies
# IMPORTANT: Verify the license is MIT, BSD, Apache, or ISC — never GPL/AGPL

# Add a dev dependency
# Edit pyproject.toml -> [project.optional-dependencies] dev

# Reinstall after changes
make install
```

### Database Migrations

```bash
# Generate a new migration
alembic -c migrations/alembic.ini revision --autogenerate -m "mlo_add_experiments_table"

# Apply migrations
alembic -c migrations/alembic.ini upgrade head

# Roll back one migration
alembic -c migrations/alembic.ini downgrade -1
```

### Docker

```bash
# Build image
make docker-build

# Start all services (app + postgres + redis + kafka + mlflow)
docker compose -f docker-compose.dev.yml up -d

# View logs
docker compose -f docker-compose.dev.yml logs -f app
```

## Related Repos

| Repo | Relationship | Description |
|------|-------------|-------------|
| [aumos-common](https://github.com/aumos-enterprise/aumos-common) | Dependency | Shared utilities, auth, database, events |
| [aumos-proto](https://github.com/aumos-enterprise/aumos-proto) | Dependency | Protobuf event schemas |
| [aumos-model-registry](https://github.com/aumos-enterprise/aumos-model-registry) | Upstream | Model versions and metadata consumed for deployment |
| [aumos-drift-detector](https://github.com/aumos-enterprise/aumos-drift-detector) | Downstream | Consumes deployment events to begin drift monitoring |
| [aumos-observability](https://github.com/aumos-enterprise/aumos-observability) | Downstream | Ingests experiment metrics and deployment health |
| [aumos-llm-serving](https://github.com/aumos-enterprise/aumos-llm-serving) | Downstream | Receives deployment instructions for LLM inference |
| [aumos-agent-framework](https://github.com/aumos-enterprise/aumos-agent-framework) | Downstream | Uses feature store APIs to enrich agent context |

## License

Copyright 2026 AumOS Enterprise. Licensed under the [Apache License 2.0](LICENSE).

This software must not incorporate AGPL or GPL licensed components.
See [CONTRIBUTING.md](CONTRIBUTING.md) for license compliance requirements.
