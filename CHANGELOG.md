# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-02-26

### Added

- Initial project scaffolding from aumos-repo-template
- FastAPI application with lifespan management, liveness and readiness probes
- `Settings` class extending `AumOSSettings` with `AUMOS_MLOPS_` env prefix
- SQLAlchemy ORM models with `mlo_` table prefix:
  - `Experiment` — MLflow-backed experiment with tenant isolation
  - `Deployment` — Model deployment with canary/A-B/shadow/blue-green strategy support
  - `FeatureSet` — Feast-backed feature set definition with JSONB feature schema
  - `RetrainingJob` — Drift-triggered, scheduled, and manual retraining job tracking
- Protocol interfaces for all repository types (`IExperimentRepository`, `IDeploymentRepository`, `IFeatureSetRepository`, `IRetrainingJobRepository`)
- Service layer:
  - `ExperimentService` — create, list, log_run, compare_runs
  - `DeploymentService` — deploy, rollback, promote, get_status, canary_progress
  - `FeatureStoreService` — create_feature_set, get_features, materialize
  - `RetrainingService` — trigger, get_status, schedule
- REST API endpoints for experiments, deployments, feature sets, and retraining jobs
- `MLflowClient` adapter for tenant-isolated MLflow tracking
- `FeastClient` adapter for feature store operations
- `MLOpsEventPublisher` with typed publish methods for all lifecycle events
- SQLAlchemy repositories extending `BaseRepository` from aumos-common
- CI/CD pipeline (lint, typecheck, test, docker build, license check)
- Docker multi-stage build with non-root `aumos` user
- `docker-compose.dev.yml` with PostgreSQL, Redis, Kafka, and MLflow services
- Apache 2.0 license, CONTRIBUTING.md with GPL/AGPL prohibition
- SECURITY.md with responsible disclosure policy to security@aumos.io
