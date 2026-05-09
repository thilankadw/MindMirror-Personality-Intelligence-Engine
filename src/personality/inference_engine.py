"""Utilities for inference engine."""
from typing import List, Optional

from src.inference.artifact_resolver import ArtifactResolver
from src.inference.base_engine import BaseInferenceEngine
from src.inference.schema_registry import InferenceSchemaRegistry


class PersonalityInferenceEngine(BaseInferenceEngine):
    """Big Five trait inference engine backed by the shared base engine."""

    def __init__(
        self,
        trait: str,
        model_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        artifact_resolver: Optional[ArtifactResolver] = None,
        schema_registry: Optional[InferenceSchemaRegistry] = None,
    ):
        """Initialize the personality inference engine."""
        if not trait:
            raise ValueError("trait is required for personality inference.")

        super().__init__(
            domain="personality",
            trait=trait,
            model_path=model_path,
            feature_columns=feature_columns,
            target_features=target_features,
            artifact_resolver=artifact_resolver,
            schema_registry=schema_registry,
        )
