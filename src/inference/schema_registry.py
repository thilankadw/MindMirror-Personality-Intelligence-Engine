"""Inference utilities for schema registry."""
from dataclasses import dataclass
from typing import List, Optional

from utils.config import (
    get_configured_feature_columns,
    get_mean_features,
    get_std_features,
    get_target_features,
    get_temporal_features,
)


@dataclass(frozen=True)
class InferenceSchema:
    """Represent inference schema."""
    feature_columns: List[str]
    target_features: List[str]


class InferenceSchemaRegistry:
    """Resolve feature and target schemas for inference engines."""

    def resolve(
        self,
        domain: str = "personality",
        trait: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
    ) -> InferenceSchema:
        """Resolve the requested data."""
        resolved_features = (
            list(feature_columns)
            if feature_columns is not None
            else self._resolve_feature_columns(domain=domain, trait=trait)
        )
        resolved_targets = (
            list(target_features)
            if target_features is not None
            else get_target_features(domain=domain, trait=trait)
        )
        return InferenceSchema(
            feature_columns=resolved_features,
            target_features=resolved_targets,
        )

    def _resolve_feature_columns(
        self,
        domain: str,
        trait: Optional[str] = None,
    ) -> List[str]:
        """Resolve feature columns."""
        configured_columns = get_configured_feature_columns(domain=domain, trait=trait)
        if configured_columns:
            return configured_columns

        if domain == "personality":
            return (
                get_mean_features(domain=domain, trait=trait)
                + get_std_features(domain=domain, trait=trait)
                + get_temporal_features(domain=domain, trait=trait)
            )

        return get_temporal_features(domain=domain, trait=trait)
