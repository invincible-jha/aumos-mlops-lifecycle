# CLAUDE.md — AumOS MLOps Lifecycle

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-mlops-lifecycle`) is part of **Tier 3: Intelligence Operations**:
end-to-end MLOps/LLMOps lifecycle management enabling experiment tracking, model deployment,
feature engineering, and automated retraining at enterprise scale.

**Release Tier:** B: Open Core
**Product Mapping:** Product 7 — MLOps & LLMOps Platform
**Phase:** 3 (Months 12-18)

## Repo Purpose

`aumos-mlops-lifecycle` is the central orchestration service for the AumOS MLOps platform.
It provides experiment tracking via MLflow, zero-downtime model deployment with canary/A-B/shadow
and blue-green strategies, feature store management via Feast, and automated model retraining
triggered by drift detection signals. Every ML model running in production on AumOS passes
through this service's deployment and monitoring pipelines.

## Architecture Position

```
aumos-common ──────────────────────────────────────────────────────►
aumos-proto  ──────────────────────────────────────────────────────►
aumos-model-registry ──────────────────────────────────────────────► aumos-mlops-lifecycle ──► aumos-drift-detector (consumes events)
                                                                                           ──► aumos-observability (metrics/traces)
                                                                                           ──► aumos-event-bus (publishes MLOps events)
                                                                                           ──► aumos-data-layer (stores job state)
                                                                                           ──► aumos-llm-serving (deployment targets)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-model-registry` — Model versions and metadata for deployment

**Downstream dependents (other repos IMPORT from this):**
- `aumos-drift-detector` — Consumes deployment events to begin drift monitoring
- `aumos-observability` — Ingests experiment run metrics and deployment health signals
- `aumos-llm-serving` — Receives deployment instructions for LLM inference endpoints
- `aumos-agent-framework` — Uses feature store APIs to enrich agent context

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| mlflow | 2.10+ | Experiment tracking and model registry client |
| feast | 0.37+ | Feature store management and materialization |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app

   # WRONG — never reimplement these
   # from jose import jwt  (use aumos_common.auth instead)
   # from sqlalchemy import create_engine  (use aumos_common.database instead)
   # import logging  (use aumos_common.observability.get_logger instead)
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.** Use MLOps-specific topics.

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure Convention

```
src/aumos_mlops_lifecycle/
├── __init__.py                    # __version__ = "0.1.0"
├── main.py                        # FastAPI app with create_app(), lifespan, health checks
├── settings.py                    # Extends AumOSSettings, env prefix AUMOS_MLOPS_
├── api/
│   ├── __init__.py
│   ├── router.py                  # All API endpoints for experiments, deployments, feature sets, retraining
│   └── schemas.py                 # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy ORM models (mlo_ prefix)
│   ├── interfaces.py              # Protocol interfaces for all repositories
│   └── services.py                # ExperimentService, DeploymentService, FeatureStoreService, RetrainingService
└── adapters/
    ├── __init__.py
    ├── repositories.py            # ExperimentRepository, DeploymentRepository, FeatureSetRepository, RetrainingJobRepository
    ├── kafka.py                   # MLOpsEventPublisher with typed publish methods
    ├── mlflow_client.py           # MLflow tracking client wrapper
    └── feast_client.py            # Feast feature store client wrapper
```

## API Conventions

- All endpoints under `/api/v1/` prefix
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Request ID: `X-Request-ID` header (auto-generated if missing)
- Pagination: `?page=1&page_size=20&sort_by=created_at&sort_order=desc`
- Errors: Standard `ErrorResponse` from aumos-common
- Content-Type: `application/json` (always)

### Endpoint Map

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/experiments` | Create experiment |
| GET | `/api/v1/experiments` | List experiments |
| GET | `/api/v1/experiments/{id}` | Get experiment |
| POST | `/api/v1/experiments/{id}/runs` | Log experiment run |
| GET | `/api/v1/experiments/{id}/runs` | List runs for experiment |
| POST | `/api/v1/deployments` | Create deployment |
| GET | `/api/v1/deployments` | List deployments |
| GET | `/api/v1/deployments/{id}` | Get deployment |
| POST | `/api/v1/deployments/{id}/rollback` | Rollback deployment |
| POST | `/api/v1/feature-sets` | Create feature set |
| GET | `/api/v1/feature-sets` | List feature sets |
| GET | `/api/v1/feature-sets/{id}` | Get feature set |
| POST | `/api/v1/retraining-jobs` | Trigger retraining job |
| GET | `/api/v1/retraining-jobs` | List retraining jobs |
| GET | `/api/v1/retraining-jobs/{id}` | Get retraining job status |

## Database Conventions

- Table prefix: `mlo_` (e.g., `mlo_experiments`, `mlo_deployments`)
- ALL tenant-scoped tables: extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)
- Migration naming: `{timestamp}_mlo_{description}.py`
- Foreign keys to other repos' tables: use UUID type, no FK constraints (cross-service)
- JSONB columns used for: tags, features list, deployment metrics, retraining metrics

### ORM Model Summary

| Model | Table | Key Fields |
|-------|-------|-----------|
| `Experiment` | `mlo_experiments` | name, description, status, mlflow_experiment_id, tags |
| `Deployment` | `mlo_deployments` | model_id, model_version, strategy, status, target_environment, traffic_split, health_check_url |
| `FeatureSet` | `mlo_feature_sets` | name, entity_name, features (JSONB), source_type, schedule |
| `RetrainingJob` | `mlo_retraining_jobs` | model_id, trigger_type, status, started_at, completed_at, metrics (JSONB) |

### Deployment Strategy Values

- `canary` — Gradual traffic shift with automatic rollback on error threshold
- `ab` — A/B testing with traffic split (traffic_split = {"v1": 50, "v2": 50})
- `shadow` — Mirror traffic to new model without serving responses
- `blue-green` — Instant cutover with pre-warmed standby environment

### Retraining Trigger Types

- `drift` — Triggered by aumos-drift-detector publishing a drift event
- `scheduled` — Cron-based periodic retraining
- `manual` — Operator-initiated via API

## Kafka Events Published

All events published by this service to Kafka:

| Topic Constant | Event | Payload Fields |
|----------------|-------|----------------|
| `Topics.MLO_EXPERIMENT_CREATED` | Experiment created | tenant_id, experiment_id, name |
| `Topics.MLO_RUN_LOGGED` | Run logged to experiment | tenant_id, experiment_id, run_id, metrics |
| `Topics.MLO_DEPLOYMENT_CREATED` | Deployment initiated | tenant_id, deployment_id, model_id, model_version, strategy |
| `Topics.MLO_DEPLOYMENT_COMPLETED` | Deployment finished | tenant_id, deployment_id, status, target_environment |
| `Topics.MLO_DEPLOYMENT_ROLLED_BACK` | Rollback executed | tenant_id, deployment_id, reason |
| `Topics.MLO_FEATURE_SET_CREATED` | Feature set created | tenant_id, feature_set_id, name |
| `Topics.MLO_RETRAINING_TRIGGERED` | Retraining job started | tenant_id, job_id, model_id, trigger_type |
| `Topics.MLO_RETRAINING_COMPLETED` | Retraining job finished | tenant_id, job_id, status, metrics |

## Repo-Specific Context

### MLflow Integration

- `adapters/mlflow_client.py` wraps the MLflow Python client (Apache 2.0 license, safe)
- Each tenant gets an isolated MLflow experiment namespace: `tenant_{tenant_id}/{experiment_name}`
- Experiment IDs from MLflow are stored in `mlo_experiments.mlflow_experiment_id` for cross-referencing
- Run metrics are logged through the MLflow client but also persisted to PostgreSQL for SQL queries

### Feast Integration

- `adapters/feast_client.py` wraps the Feast Python client (Apache 2.0 license, safe)
- Feature sets map to Feast `FeatureView` objects; materialization is triggered via Feast SDK
- `mlo_feature_sets.features` is JSONB containing the list of feature definitions
- Schedule field uses cron syntax (e.g., `"0 */6 * * *"` for every 6 hours)

### Canary Deployment Logic

- `traffic_split` is a JSONB dict: `{"stable": 90, "canary": 10}`
- `DeploymentService.canary_progress()` increments canary traffic by a configured step (default: 10%)
- Auto-rollback: if error rate exceeds `AUMOS_MLOPS_CANARY_ERROR_THRESHOLD` during progression, rollback is triggered
- `DeploymentService.promote()` moves canary to 100% and retires the stable version

### Performance Requirements

- Experiment creation: < 200ms p99 (MLflow call is async)
- Deployment status check: < 50ms p99 (reads from local DB, not remote)
- Feature materialization: async background task, not in request path
- Retraining trigger: < 100ms p99 (enqueues job, returns immediately)

### What Claude Code Should NOT Do in This Repo

1. Do NOT call MLflow APIs synchronously in request handlers — always use the async wrapper
2. Do NOT store raw ML model artifacts in PostgreSQL — store only references (MLflow run IDs, artifact URIs)
3. Do NOT expose Feast internals in API responses — translate to `FeatureSetResponse` Pydantic models
4. Do NOT implement drift detection logic here — that lives in `aumos-drift-detector`
5. Do NOT hardcode MLflow server URL — use `AUMOS_MLOPS_MLFLOW_TRACKING_URI` env var
6. Do NOT hardcode Feast registry path — use `AUMOS_MLOPS_FEAST_REGISTRY_PATH` env var
7. Do NOT skip tenant isolation on MLflow experiment creation — every experiment must be namespaced

## Testing Requirements

- Minimum coverage: **80%** for `core/` (services, models, interfaces)
- Minimum coverage: **60%** for `adapters/` (repositories, kafka, mlflow_client, feast_client)
- Mock MLflow and Feast clients in unit tests using `unittest.mock.AsyncMock`
- Use `testcontainers` for PostgreSQL integration tests
- Use `pytest-asyncio` with `asyncio_mode = "auto"` (already configured in pyproject.toml)

## Environment Variables

All standard env vars are defined in `aumos_common.config.AumOSSettings`.
Repo-specific vars use the prefix `AUMOS_MLOPS_`.

See `.env.example` for the full list of required and optional variables.
