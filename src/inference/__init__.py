"""Shared inference architecture components."""

from src.inference.artifact_resolver import (
    ArtifactResolver,
    FallbackArtifactResolver,
    LocalArtifactResolver,
    RemoteArtifactResolver,
)
from src.inference.base_engine import BaseInferenceEngine
from src.inference.schema_registry import InferenceSchema, InferenceSchemaRegistry

__all__ = [
    "ArtifactResolver",
    "BaseInferenceEngine",
    "FallbackArtifactResolver",
    "InferenceSchema",
    "InferenceSchemaRegistry",
    "LocalArtifactResolver",
    "RemoteArtifactResolver",
]
