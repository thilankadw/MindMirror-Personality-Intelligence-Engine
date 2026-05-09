"""Utilities for model inference."""
from typing import List, Optional

from src.inference.artifact_resolver import ArtifactResolver
from src.inference.base_engine import BaseInferenceEngine
from src.inference.schema_registry import InferenceSchemaRegistry


class ModelInference(BaseInferenceEngine):
    """Backward-compatible alias over the shared inference engine."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        domain: str = "personality",
        trait: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        artifact_resolver: Optional[ArtifactResolver] = None,
        schema_registry: Optional[InferenceSchemaRegistry] = None,
    ):
        """Initialize the model inference."""
        super().__init__(
            domain=domain,
            trait=trait,
            model_path=model_path,
            feature_columns=feature_columns,
            target_features=target_features,
            artifact_resolver=artifact_resolver,
            schema_registry=schema_registry,
        )
