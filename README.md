# MindMirror Personality Inference

This repository is the personality-only slice of MindMirror. It focuses on Reddit-based Big Five personality inference, trait-specific training pipelines, MLflow-backed model loading, signup inference orchestration, and the FastAPI/runtime pieces needed to serve personality predictions.

## Scope

Kept in this repository:

- personality training and inference pipelines under `pipelines/personality/`
- core personality feature engineering and inference code under `src/personality/`
- Reddit extraction, signup worker, auth, jobs, and personality prediction API routes
- shared utilities needed by the personality runtime such as MLflow, config loading, and database access

Removed from this repository:

- cognitive, motivation, and emotion domain code
- analytics and social API surfaces
- non-personality DB models, schemas, repositories, services, scripts, and tests

## Runtime Shape

- `api`: FastAPI app for auth, Reddit linking, extraction, jobs, and personality prediction retrieval
- `worker`: database-polling signup inference worker for personality jobs
- `postgres`: external database shared by API and worker
- `mlflow`: model registry and artifact source for personality inference
- `reddit extraction`: live Reddit data collection for linked users

## Main Paths

```text
api/         FastAPI application, DB models, repositories, services, routes
pipelines/   Big Five trait training and inference entrypoints
scripts/     personality-oriented helper scripts
src/         personality logic, extraction logic, inference engines, worker logic
tests/       remaining personality/extractor/auth tests
utils/       configuration, MLflow, and model-loading utilities
```

## Key Endpoints

- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/reddit/signup`
- `POST /api/v1/users/me/reddit`
- `GET /api/v1/inference/jobs/{job_id}`
- `GET /api/v1/users/me/predictions/personality`
- `GET /api/v1/users/me/predictions`
- `GET /health`

Swagger UI:

- `http://localhost:8000/api/v1/docs`

## Personality Data Flow

1. Extract Reddit posts and comments for a linked user.
2. Build `processed_text` from titles and bodies.
3. Score post-level OCEAN traits with the Big Five transformer model when needed.
4. Generate BERT embeddings when needed.
5. Aggregate post-level features into weekly user-level rows.
6. Build trait-direction labels such as `up`, `down`, and `neutral`.
7. Train one classifier per Big Five trait.
8. Load production personality models from MLflow for inference.
9. Persist weekly personality predictions for signup jobs.

## Development Notes

- Signup inference is database-backed; Kafka support and related config were removed from this repository copy.
- The worker still supports the static demo Reddit username `ok_celery_4705` if `src/data/data.json` is present.
- The remaining config in `config.yaml` is filtered to the personality domain only.
