"""
Utility functions for resolving pretrained model and tokenizer paths.

Checks for a locally cached copy under `pretrained_models/<model_slug>/` first;
if not found, falls back to the HuggingFace hub name so that the `transformers`
library can download it automatically.

Directory layout (optional local cache):
    pretrained_models/
        roberta-base/           <- model files for roberta-base
        microsoft-deberta-v3-base/
        ...
"""
import os
import logging

logger = logging.getLogger(__name__)

# Root of the workspace (one level above utils/)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_MODELS_DIR = os.path.join(_WORKSPACE_ROOT, "pretrained_models")


def _model_slug(model_name: str) -> str:
    """Convert a HuggingFace model name to a filesystem-safe folder name."""
    return model_name.replace("/", "-")


def _local_path(model_name: str) -> str | None:
    """
    Return the local directory path for the model if it exists, else None.
    Checks:
      1. pretrained_models/<slug>/           (project-level cache)
      2. pretrained_models/<original_name>/  (exact sub-path, e.g. "microsoft/deberta-v3-base")
    """
    slug = _model_slug(model_name)

    candidates = [
        os.path.join(_LOCAL_MODELS_DIR, slug),
        os.path.join(_LOCAL_MODELS_DIR, model_name),
    ]

    for path in candidates:
        if os.path.isdir(path):
            # Verify it contains at least one model/config file
            contents = os.listdir(path)
            has_model = any(
                f.endswith((".bin", ".safetensors", ".json", ".pt"))
                for f in contents
            )
            if has_model:
                logger.info(f"Found local model at: {path}")
                return path

    return None


def get_pretrained_model_path(model_name: str) -> str | None:
    """
    Resolve the path for loading a pretrained transformer backbone.

    Args:
        model_name: HuggingFace model identifier (e.g. "roberta-base").

    Returns:
        Local directory path if a local copy exists, otherwise None
        (callers should then use `model_name` directly with HuggingFace).
    """
    local = _local_path(model_name)
    if local:
        return local

    logger.info(
        f"No local copy found for '{model_name}'. "
        "Will download from HuggingFace hub on first use."
    )
    return None


def get_pretrained_tokenizer_path(model_name: str) -> str:
    """
    Resolve the path for loading a pretrained tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. "roberta-base").

    Returns:
        Local directory path if a local copy exists, otherwise the original
        model_name string (HuggingFace hub download).
    """
    local = _local_path(model_name)
    if local:
        return local

    logger.info(
        f"No local tokenizer found for '{model_name}'. "
        "Will download from HuggingFace hub on first use."
    )
    return model_name  # AutoTokenizer.from_pretrained accepts hub names directly
