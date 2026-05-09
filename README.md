# MindMirror Personality

MindMirror Personality is the personality-only slice of the MindMirror system. It trains, registers, serves, and persists Big Five weekly personality direction predictions derived from Reddit activity.

The repository now focuses on one domain only:

- `personality`

The active production path is:

1. A user links a Reddit username through the API.
2. The API creates a signup inference job in Postgres.
3. The worker polls queued jobs from the database.
4. Reddit posts are loaded from a static demo bundle or extracted live.
5. Raw posts are normalized into weekly trait and embedding features.
6. Five per-trait inference models predict weekly direction states.
7. Results are stored in the database and exposed through prediction endpoints.

## What This Repo Contains

This codebase combines four concerns in one repository:

- `API application`: FastAPI service for authentication, user profile data, job status, and prediction retrieval.
- `ML pipelines`: trait-specific data preparation, training, and offline inference entry points.
- `Inference runtime`: shared inference engine, feature builders, artifact loading, and signup prediction orchestration.
- `MLOps and deployment`: MLflow integration, Docker packaging, compose-based local orchestration, SQL bootstrap, and operational scripts.

## Domain Scope

The repository intentionally excludes the older non-personality domains. Everything active here is built around the Big Five traits:

- `openness`
- `conscientiousness`
- `extraversion`
- `agreeableness`
- `neuroticism`

Each trait has its own training, artifact paths, and MLflow registry mapping in `config.yaml`.

## High-Level Architecture

### Online serving flow

`client -> FastAPI -> Postgres job row -> worker poll loop -> Reddit extraction -> feature building -> MLflow model loading -> prediction persistence -> API readback`

### Offline ML flow

`raw Reddit posts -> preprocessing -> weekly aggregation -> delta target labeling -> train/test split -> trait model training -> evaluation -> MLflow logging -> MLflow model registry`

### Runtime components

- `api`: request handling, auth, persistence, job orchestration
- `worker`: signup inference background consumer
- `src/personality`: preprocessing, features, training, evaluation, inference
- `src/inference`: shared model loading and schema logic
- `utils`: configuration and MLflow helpers
- `pipelines/personality/*`: trait entry points for data prep, training, inference

## Machine Learning Design

### Prediction task

This repository predicts trait direction state, not raw trait score regression. For each Big Five trait, the model predicts a weekly directional label derived from future-vs-past movement over a rolling time window.

### Input data assumptions

The pipeline is built around Reddit post data with fields such as:

- author or equivalent username column
- timestamp (`created_date`, `created_at`, or `created_utc`)
- text fields (`title`, `selftext`)
- optional precomputed Big Five scores
- optional precomputed embeddings

When upstream signals are missing, the runtime can compute them on demand:

- Big Five text scores from the scoring model
- BERT-style text embeddings from the embedding model

### Preprocessing stages

The preprocessing code in `src/personality` and the trait data pipelines follows this sequence:

1. Normalize input columns and author identity.
2. Concatenate `title` and `selftext` into a processed text field.
3. Clean text and replace emoji textually.
4. Parse or generate Big Five scores.
5. Parse or generate text embeddings.
6. Aggregate post-level data into user-week records.
7. Compute week gaps and temporal spacing features.
8. Fill standard deviation gaps safely for sparse weeks.

### Weekly aggregation

The system converts post-level Reddit data into weekly user summaries:

- weekly trait means
- weekly trait standard deviations
- weekly embedding means
- week gap features

This is the bridge between raw social content and model-ready longitudinal data.

### Target engineering

Training labels are generated from temporal movement in each trait:

- `past_mean` is computed over the configured lookback window
- `future_mean` is computed over the configured forward window
- `delta_future` captures the future-minus-past change
- a quantile threshold `tau` converts the delta into directional classes

Those labels become the trait-specific targets such as `openness_dir`.

### Feature engineering

The main feature families are:

- temporal gap features: `time_gap_weeks`, `log_time_gap_weeks`
- lag features: current observation shifted to prior-week context
- rolling features: rolling mean and rolling std over prior weeks
- per-week dispersion features: trait weekly std
- shifted embedding vector features: prior weekly embedding representation

At inference time, the system drops rows that do not have enough historical context. In practice, at least two weekly observations are needed to produce lagged predictions.

### Model training

The default configured model type is `xgboost`. The training stack also keeps compatibility helpers for:

- `random_forest`
- `hist_gradient_boosting`

Trait training loads or creates dataset splits, trains a classifier, evaluates it, saves artifacts locally, and registers the trained model in MLflow.

### Inference behavior

Online inference is registry-first:

- models are resolved from MLflow registry URIs such as `models:/.../Production`
- local model paths are intentionally disabled for the shared runtime inference engine
- label encoders are downloaded from MLflow model artifacts when available

This keeps runtime serving aligned with promoted registry versions rather than ad hoc local files.

## Personality Pipelines

Each trait under `pipelines/personality/<trait>/` follows the same pattern:

- `data_pipeline.py`: builds or loads trait-specific train/test datasets
- `train_pipeline.py`: trains and evaluates the model, then registers it in MLflow
- `inference_pipeline.py`: command-line wrapper over shared inference for one trait

The shared offline inference entry point is `pipelines/personality/run_trait_inference.py`.

### Data pipeline responsibilities

The trait data pipelines do the heavy transformation work:

- ingest raw JSON or CSV
- detect whether the input is already weekly
- preprocess text if needed
- parse score payloads into trait columns
- generate embeddings if not already present
- aggregate post data by user-week
- compute future delta labels
- create lag and rolling features
- split by author and time
- persist `X_train`, `X_test`, `Y_train`, `Y_test`
- log dataset metadata and thresholds to MLflow

### Training pipeline responsibilities

The trait training pipelines:

- ensure split artifacts exist
- instantiate the configured model builder
- train the model
- evaluate it on the held-out split
- persist evaluation reports
- log metrics, datasets, parameters, and artifacts to MLflow
- optionally save and log a label encoder
- register the trained model version in MLflow Model Registry

### Inference pipeline responsibilities

The trait inference wrappers:

- load posts from CSV, JSON, Python data structures, or DataFrames
- build inference features from raw Reddit posts
- resolve the correct registered model
- make predictions
- optionally export JSON or CSV prediction outputs

## Serving and Application Flow

### FastAPI application

The API app is created in `api/app/main.py`. On startup it:

- configures structured logging
- sets CORS
- installs request ID middleware
- optionally creates database tables
- mounts the versioned API router
- exposes `/health`

### Main API routes

- `POST /api/v1/users/me/reddit`
  Saves a Reddit username, queues signup inference, and optionally waits for completion.
- `GET /api/v1/inference/jobs/{job_id}`
  Returns queued, failed, or completed job state.
- `GET /api/v1/users/me/predictions/personality`
  Returns the latest personality inference run for the current user.
- `GET /api/v1/users/me/predictions`
  Returns all available domain payloads. In this repo that means personality only.
- `GET /api/v1/users/me`
  Returns the current user plus linked Reddit username and latest signup inference timestamp.

Legacy extraction and caption routes are still present in `jobs.py`, but the core product path is the signup inference route plus prediction retrieval.

### Signup inference flow

The end-to-end signup path spans multiple modules:

1. `users.py` accepts the Reddit username.
2. `signup_inference_service.py` upserts the Reddit identity and creates an inference job.
3. `src/workers/signup_inference_consumer.py` polls the next queued job.
4. `src/services/reddit_extractor.py` loads cached, static, or live Reddit post bundles.
5. `src/services/personality_signup_inference.py` builds features and runs all five traits.
6. `prediction_repository.py` persists the run header and weekly trait predictions.
7. `predictions.py` reads the latest run back for clients.

### Reddit extraction

The live extraction path uses `src/extractors/reddit.py` through `src/services/reddit_extractor.py`.

Notable behavior:

- static demo user `ok_celery_4705` reads from a committed JSON bundle
- cached bundles are reused from `artifacts/extractions/...`
- image URLs can be downloaded locally
- BLIP captioning can enrich media summaries when vision dependencies are available

## Configuration

`config.yaml` is the central project configuration.

It defines:

- project metadata
- service naming
- database settings
- artifact locations
- active domains
- trait-specific data paths
- training settings
- model hyperparameters
- MLflow tracking and registry mappings
- inference loading stage

### Important configuration concepts

- `domains.personality.traits.<trait>`
  Holds per-trait paths and direction-label settings.
- `mlflow.models`
  Maps model keys like `big5_openness_state` to registry model names.
- `inference_loading.model_stage`
  Controls which MLflow stage the runtime uses, unless `MODEL_STAGE` overrides it.

## MLflow and MLOps

This repository treats MLflow as both an experiment tracker and the serving registry boundary.

### What is logged

Depending on the pipeline stage, MLflow captures:

- dataset sizes
- feature names
- train/test split stats
- label threshold `tau`
- model hyperparameters
- evaluation metrics
- saved model artifacts
- label encoder artifacts
- pipeline metadata JSON

### Registry usage

The training pipelines register trait models with model keys such as:

- `big5_openness_state`
- `big5_conscientiousness_state`
- `big5_extraversion_state`
- `big5_agreeableness_state`
- `big5_neuroticism_state`

The inference engine resolves those model keys into registry URIs and loads them from the configured stage, typically `Production`.

### Runtime tracking behavior

`utils/mlflow_utils.py` and `utils/mlflow_registry.py` provide:

- tracking URI setup
- file-store fallback support
- experiment bootstrap
- run naming and tagging
- registry model registration helpers
- registry URI validation

### MLOps posture in this repository

This is not a full enterprise orchestration stack, but it does include practical MLOps building blocks:

- versioned configuration
- reproducible train/test split artifacts
- metrics and artifact tracking
- model registry promotion boundary
- containerized serving
- compose-based local orchestration
- SQL bootstrap for persistence

Airflow and Kafka compose overlays are referenced by scripts if those files exist, but they are optional and not required for the core personality flow.

## Deployment and Containerization

### Dockerfile

The root `Dockerfile`:

- uses `python:3.11-slim`
- installs system build dependencies
- creates a virtual environment at `/opt/venv`
- installs Python dependencies from `requirements.txt`
- copies `api`, `src`, `utils`, `pipelines`, and `config.yaml`
- starts the API with Uvicorn on port `8000`

### docker-compose.yml

The active compose file defines two services:

- `api`
  Runs the FastAPI application and exposes port `8000`.
- `worker`
  Runs the signup inference consumer loop.

Both services mount the project code so local changes are reflected in the containers. The worker also installs `demoji` defensively before starting.

### Database

The repository is designed to work with Postgres through SQLAlchemy async sessions. SQL bootstrap details live under `sql/`.

### Deployment notes

- API and worker are logically separate processes.
- The worker currently polls the database instead of consuming an external queue.
- Model serving depends on access to the configured MLflow tracking and registry backend.
- `CREATE_TABLES_ON_STARTUP` can bootstrap tables for development, but real deployments should prefer migrations or SQL bootstrap scripts.

## Local Development

### Python

- Target Python: `3.11`
- Package metadata is declared in `pyproject.toml`

### Typical local flow

1. Create and activate a Python 3.11 environment.
2. Install dependencies.
3. Configure `.env`.
4. Ensure Postgres and MLflow are reachable.
5. Start the API and worker.

### Scripts

- `scripts/run_local.sh`
  Brings up the compose stack, optionally layering Kafka and Airflow compose files if they exist.
- `scripts/up_full_stack.sh`
  Same idea, with service URLs echoed after startup.
- `scripts/down_full_stack.sh`
  Stops the compose stack.
- `scripts/migrate_db.sh`
  Applies database migration/bootstrap steps.
- `scripts/extract_reddit_posts.py`
  Utility for collecting Reddit posts outside the API path.
- `scripts/export_signup_flow_predictions.py`
  Exports persisted signup inference predictions.
- `scripts/save_signup_predictions_direct.py`
  Saves signup predictions directly into storage for backfill or debugging flows.

## SQL and Persistence

The current SQL bootstrap notes are in `sql/README.md`.

`sql/001_init.sql` initializes the main persistence objects used by the app:

- users
- jobs
- predictions
- related uniqueness constraints

At the ORM layer, the app also maintains models for:

- Reddit-linked platform identities
- inference runs
- weekly predictions
- posts, comments, and media metadata

## Repository Structure

```text
api/         FastAPI app, DB models, repositories, schemas, services
pipelines/   Trait-specific offline data/train/inference entry points
src/         Shared ML, extraction, and worker runtime code
scripts/     Local orchestration and data utility scripts
sql/         SQL bootstrap assets
tests/       Focused extractor and auth tests
utils/       Central config and MLflow helpers
```

## File-by-File Guide

The list below reflects the current personality-only repository contents that matter to the runtime and ML pipelines.

### Root files

| File | Purpose |
| --- | --- |
| `README.md` | Project documentation and architecture guide. |
| `config.yaml` | Central configuration for personality traits, artifacts, MLflow, and inference loading. |
| `Dockerfile` | API container image definition. |
| `docker-compose.yml` | Local multi-service runtime for API plus worker. |
| `pyproject.toml` | Python package metadata, dependencies, and tooling configuration. |

### `api/app`

| File | Purpose |
| --- | --- |
| `api/app/__init__.py` | Marks the FastAPI app package. |
| `api/app/__main__.py` | CLI/module entry point for the packaged app. |
| `api/app/main.py` | Builds the FastAPI app, middleware, startup DB initialization, and health endpoint. |

### `api/app/api`

| File | Purpose |
| --- | --- |
| `api/app/api/__init__.py` | API package marker. |
| `api/app/api/deps.py` | Shared dependency helpers for authenticated route access. |
| `api/app/api/router.py` | Mounts auth, jobs, users, and predictions routers. |

### `api/app/api/v1`

| File | Purpose |
| --- | --- |
| `api/app/api/v1/__init__.py` | Versioned API package marker. |

### `api/app/api/v1/routes`

| File | Purpose |
| --- | --- |
| `api/app/api/v1/routes/auth.py` | Authentication endpoints and token-related flows. |
| `api/app/api/v1/routes/jobs.py` | Extraction, captioning, and job-status endpoints. |
| `api/app/api/v1/routes/predictions.py` | Endpoints for latest personality predictions. |
| `api/app/api/v1/routes/users.py` | Current-user profile endpoints plus signup inference trigger. |

### `api/app/core`

| File | Purpose |
| --- | --- |
| `api/app/core/__init__.py` | Core package marker. |
| `api/app/core/config.py` | Pydantic settings and environment-backed runtime configuration. |
| `api/app/core/logging.py` | Logging setup utilities. |
| `api/app/core/security.py` | JWT and current-user security helpers. |

### `api/app/clients`

| File | Purpose |
| --- | --- |
| `api/app/clients/__init__.py` | Reserved package marker for external client integrations. |

### `api/app/db`

| File | Purpose |
| --- | --- |
| `api/app/db/__init__.py` | Database package marker. |
| `api/app/db/base.py` | Shared SQLAlchemy declarative base. |
| `api/app/db/bootstrap_schema.py` | Database schema bootstrap helpers. |
| `api/app/db/deps.py` | FastAPI database-session dependency functions. |
| `api/app/db/session.py` | Async engine and session factory creation. |

### `api/app/db/models`

| File | Purpose |
| --- | --- |
| `api/app/db/models/__init__.py` | Re-exports ORM models used across the app. |
| `api/app/db/models/comment.py` | ORM model for extracted Reddit comments. |
| `api/app/db/models/job.py` | ORM model for queued and completed inference jobs. |
| `api/app/db/models/media.py` | ORM model for downloaded or described media assets. |
| `api/app/db/models/post.py` | ORM model for Reddit posts. |
| `api/app/db/models/prediction.py` | ORM models for inference runs and weekly personality predictions. |
| `api/app/db/models/user.py` | ORM model for application users. |
| `api/app/db/models/user_inference_snapshot.py` | ORM model for user-level inference snapshot state. |
| `api/app/db/models/user_platform_identity.py` | ORM model linking app users to external identities such as Reddit usernames. |

### `api/app/repositories`

| File | Purpose |
| --- | --- |
| `api/app/repositories/__init__.py` | Repository package marker. |
| `api/app/repositories/job_repository.py` | CRUD helpers for inference job state transitions and queue polling. |
| `api/app/repositories/prediction_repository.py` | Persistence and formatting helpers for personality inference runs and weekly predictions. |
| `api/app/repositories/user_repository.py` | User lookup plus platform identity helpers. |

### `api/app/schemas`

| File | Purpose |
| --- | --- |
| `api/app/schemas/__init__.py` | Schema package marker. |
| `api/app/schemas/auth.py` | Request and response schemas for authentication. |
| `api/app/schemas/job.py` | Schemas for inference job payloads and status responses. |
| `api/app/schemas/user.py` | Core user response schemas. |
| `api/app/schemas/user_reddit.py` | Reddit username input and signup inference output schemas. |

### `api/app/services`

| File | Purpose |
| --- | --- |
| `api/app/services/__init__.py` | Services package marker. |
| `api/app/services/auth_service.py` | Authentication service logic. |
| `api/app/services/job_service.py` | Wait/retry helpers for job completion payload resolution. |
| `api/app/services/reddit_oauth_service.py` | Reddit OAuth-related service utilities. |
| `api/app/services/signup_inference_service.py` | Queues personality signup inference and optionally waits for completion. |

### `api/app/services/core`

| File | Purpose |
| --- | --- |
| `api/app/services/core/__init__.py` | Core service package marker. |
| `api/app/services/core/extraction.py` | User-data extraction service abstraction used by legacy extraction routes. |
| `api/app/services/core/vision.py` | Media captioning and image-processing service used by legacy caption routes. |

### `api/app/utils`

| File | Purpose |
| --- | --- |
| `api/app/utils/__init__.py` | Utility package marker. |
| `api/app/utils/middleware.py` | Request ID middleware. |
| `api/app/utils/security.py` | Lower-level security helpers used by the app. |

### `pipelines/personality`

| File | Purpose |
| --- | --- |
| `pipelines/__init__.py` | Root pipelines package marker. |
| `pipelines/personality/__init__.py` | Personality pipeline package marker. |
| `pipelines/personality/run_trait_inference.py` | Shared offline inference loader, feature prep, and prediction exporter for one trait. |

### `pipelines/personality/agreeableness`

| File | Purpose |
| --- | --- |
| `pipelines/personality/agreeableness/__init__.py` | Trait package marker. |
| `pipelines/personality/agreeableness/data_pipeline.py` | Builds agreeableness train/test datasets and logs dataset metadata. |
| `pipelines/personality/agreeableness/inference_pipeline.py` | CLI inference wrapper for agreeableness. |
| `pipelines/personality/agreeableness/train_pipeline.py` | Trains, evaluates, and registers the agreeableness model. |

### `pipelines/personality/conscientiousness`

| File | Purpose |
| --- | --- |
| `pipelines/personality/conscientiousness/__init__.py` | Trait package marker. |
| `pipelines/personality/conscientiousness/data_pipeline.py` | Builds conscientiousness train/test datasets and logs dataset metadata. |
| `pipelines/personality/conscientiousness/inference_pipeline.py` | CLI inference wrapper for conscientiousness. |
| `pipelines/personality/conscientiousness/train_pipeline.py` | Trains, evaluates, and registers the conscientiousness model. |

### `pipelines/personality/extraversion`

| File | Purpose |
| --- | --- |
| `pipelines/personality/extraversion/__init__.py` | Trait package marker. |
| `pipelines/personality/extraversion/data_pipeline.py` | Builds extraversion train/test datasets and logs dataset metadata. |
| `pipelines/personality/extraversion/inference_pipeline.py` | CLI inference wrapper for extraversion. |
| `pipelines/personality/extraversion/train_pipeline.py` | Trains, evaluates, and registers the extraversion model. |

### `pipelines/personality/neuroticism`

| File | Purpose |
| --- | --- |
| `pipelines/personality/neuroticism/__init__.py` | Trait package marker. |
| `pipelines/personality/neuroticism/data_pipeline.py` | Builds neuroticism train/test datasets and logs dataset metadata. |
| `pipelines/personality/neuroticism/inference_pipeline.py` | CLI inference wrapper for neuroticism. |
| `pipelines/personality/neuroticism/train_pipeline.py` | Trains, evaluates, and registers the neuroticism model. |

### `pipelines/personality/openness`

| File | Purpose |
| --- | --- |
| `pipelines/personality/openness/__init__.py` | Trait package marker. |
| `pipelines/personality/openness/data_pipeline.py` | Builds openness train/test datasets and logs dataset metadata. |
| `pipelines/personality/openness/inference_pipeline.py` | CLI inference wrapper for openness. |
| `pipelines/personality/openness/train_pipeline.py` | Trains, evaluates, and registers the openness model. |

### `src/extractors`

| File | Purpose |
| --- | --- |
| `src/__init__.py` | Root source package marker. |
| `src/extractors/__init__.py` | Extractor package marker. |
| `src/extractors/base.py` | Base extractor abstractions. |
| `src/extractors/models.py` | Data models for extractor accounts and extracted records. |
| `src/extractors/reddit.py` | Reddit post extraction logic. |
| `src/extractors/registry.py` | Extractor registry and lookup helpers. |
| `src/extractors/service.py` | Higher-level service wrapper over extractor implementations. |
| `src/extractors/storage.py` | Persistence or caching helpers for extracted bundles. |

### `src/infra`

| File | Purpose |
| --- | --- |
| `src/infra/__init__.py` | Reserved package marker for infrastructure-layer helpers. |

### `src/inference`

| File | Purpose |
| --- | --- |
| `src/inference/__init__.py` | Shared inference package marker. |
| `src/inference/artifact_resolver.py` | Resolves model URIs, with MLflow registry as the supported path. |
| `src/inference/base_engine.py` | Shared MLflow-backed inference engine for loading models and predicting. |
| `src/inference/schema_registry.py` | Resolves feature and target schemas for inference scopes. |

### `src/personality`

| File | Purpose |
| --- | --- |
| `src/personality/__init__.py` | Personality ML package marker. |
| `src/personality/bigfive_scores.py` | Transformer-based Big Five scoring from text. |
| `src/personality/data_ingestion.py` | CSV and JSON dataset ingestion utilities. |
| `src/personality/data_parser.py` | Parsers for score payloads, embeddings, and related structured fields. |
| `src/personality/data_preprocess.py` | Text cleaning, column dropping, emoji replacement, and concatenation transforms. |
| `src/personality/data_splitter.py` | Author-aware longitudinal train/test splitting logic. |
| `src/personality/delta_targets.py` | Future delta target creation, label generation, and NaN handling. |
| `src/personality/feature_scaling.py` | Feature scaling utilities for model inputs. |
| `src/personality/inference_engine.py` | Personality-specific wrapper over the shared inference engine. |
| `src/personality/inference_feature_builder.py` | Builds model-ready weekly inference features directly from raw posts. |
| `src/personality/lagged_features.py` | Rolling, lag, and shift feature generators. |
| `src/personality/model_building.py` | Factory/builders for supported classifier types. |
| `src/personality/model_evaluation.py` | Evaluation metrics and report persistence. |
| `src/personality/model_inference.py` | Backward-compatible alias over the shared inference engine. |
| `src/personality/model_training.py` | Simple model fit/save/load wrapper. |
| `src/personality/preprocessing.py` | Shared inference preprocessing flow for Big Five pipelines. |
| `src/personality/smote_preprocessing.py` | Oversampling or imbalance-preparation helpers for training data. |
| `src/personality/vector_embedding.py` | BERT-style embedding generation utilities. |
| `src/personality/weekly_preprocess.py` | Weekly trait and embedding aggregation plus gap-week calculations. |

### `src/services`

| File | Purpose |
| --- | --- |
| `src/services/__init__.py` | Service package marker. |
| `src/services/domain_inference_utils.py` | Shared helpers for domain inference orchestration. |
| `src/services/personality_signup_inference.py` | Runs all five trait predictions for one signup job and persists results. |
| `src/services/reddit_extractor.py` | Loads static, cached, or live Reddit bundles and optionally enriches media summaries. |

### `src/workers`

| File | Purpose |
| --- | --- |
| `src/workers/__init__.py` | Worker package marker. |
| `src/workers/signup_inference_consumer.py` | Database-polling worker that executes queued signup inference jobs. |

### `scripts`

| File | Purpose |
| --- | --- |
| `scripts/down_full_stack.sh` | Stops the compose-based local stack. |
| `scripts/export_signup_flow_predictions.py` | Exports stored signup prediction results. |
| `scripts/extract_reddit_posts.py` | Utility script for collecting Reddit posts outside the API. |
| `scripts/migrate_db.sh` | Database migration/bootstrap helper script. |
| `scripts/run_local.sh` | Starts the local compose stack with optional overlays. |
| `scripts/save_signup_predictions_direct.py` | Direct persistence/backfill helper for signup predictions. |
| `scripts/up_full_stack.sh` | Starts the local stack and prints key service URLs. |

### `sql`

| File | Purpose |
| --- | --- |
| `sql/001_init.sql` | Initial SQL bootstrap for the persistence schema. |
| `sql/README.md` | Short notes about the SQL bootstrap file. |

### `tests`

| File | Purpose |
| --- | --- |
| `tests/test_extractors.py` | Coverage for extractor behavior. |
| `tests/test_reddit_oauth_auth.py` | Coverage for Reddit OAuth/auth-related behavior. |

### `utils`

| File | Purpose |
| --- | --- |
| `utils/__init__.py` | Utility package marker. |
| `utils/config.py` | YAML-backed configuration accessors used across API, training, and inference. |
| `utils/mlflow_registry.py` | Minimal MLflow registry URI setup and resolution helpers for serving. |
| `utils/mlflow_utils.py` | Rich MLflow experiment, logging, and registration helper utilities. |
| `utils/model_loader.py` | Generic model loading helper utilities. |

## Suggested Reading Order

If you are new to the repository, read it in this order:

1. `config.yaml`
2. `api/app/main.py`
3. `api/app/api/v1/routes/users.py`
4. `api/app/services/signup_inference_service.py`
5. `src/workers/signup_inference_consumer.py`
6. `src/services/personality_signup_inference.py`
7. `src/personality/inference_feature_builder.py`
8. `src/inference/base_engine.py`
9. one trait pipeline under `pipelines/personality/openness/`
10. `utils/mlflow_utils.py`

That path shows the system from request handling through prediction persistence and then back to training and MLOps.

## Operational Caveats

- The repository still contains some generic naming around “social” or extraction abstractions even though the active provider path is Reddit.
- The worker is queue-less in the external sense; it polls the database rather than consuming Kafka or a dedicated broker.
- Inference requires MLflow registry availability for the configured models and stage.
- Lagged weekly features mean single isolated weeks are generally not enough for full inference output.
- Legacy extraction and vision routes remain in the API, but the personality signup flow is the main maintained path.

## Summary

This repository is a personality-focused ML application that combines:

- Reddit data extraction
- Big Five feature generation
- weekly longitudinal feature engineering
- per-trait direction classification
- MLflow-based training and registry management
- FastAPI serving
- Postgres-backed job and prediction persistence
- Docker-based local deployment

If you need to extend or debug the project, start from `config.yaml`, then follow the signup inference path and one trait pipeline end to end.
