"""Utilities for MLflow registry."""
from __future__ import annotations

import os

import mlflow

from utils.config import get_inference_loading_config, get_mlflow_config, get_mlflow_model_by_key


def configure_mlflow_tracking() -> str:
    """Configure MLflow tracking."""
    tracking_uri = str(get_mlflow_config().get("tracking_uri") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def get_inference_stage() -> str:
    """Get inference stage."""
    env_stage = str(os.getenv("MODEL_STAGE") or "").strip()
    if env_stage:
        return env_stage
    configured_stage = str(get_inference_loading_config().get("model_stage") or "").strip()
    if configured_stage:
        return configured_stage
    return "Production"


def _validate_registry_uri(model_uri: str) -> str:
    """Validate registry URI."""
    uri = str(model_uri or "").strip()
    if not uri.startswith("models:/"):
        raise ValueError(
            "Only MLflow registry URIs are allowed for inference. "
            f"Received: '{model_uri}'. Expected format: 'models:/<name>/<stage>'."
        )
    return uri


def resolve_model_uri(*, model_key: str, model_uri: str | None = None, stage: str | None = None) -> str:
    """Resolve model URI."""
    if model_uri:
        return _validate_registry_uri(model_uri)

    model_cfg = get_mlflow_model_by_key(model_key) or {}
    registry_name = str(model_cfg.get("registry_name") or "").strip()
    if not registry_name:
        raise ValueError(f"No registry_name configured for model_key='{model_key}'.")

    resolved_stage = str(stage or get_inference_stage()).strip()
    if not resolved_stage:
        raise ValueError(f"Invalid model stage for model_key='{model_key}'.")
    return f"models:/{registry_name}/{resolved_stage}"
