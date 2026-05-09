"""Inference utilities for artifact resolver."""
import logging
from abc import ABC, abstractmethod
from typing import Optional

from utils.mlflow_registry import resolve_model_uri

logger = logging.getLogger(__name__)


class ArtifactResolver(ABC):
    """Resolve model URIs without coupling callers to a specific backend."""

    @abstractmethod
    def resolve_model_path(
        self,
        domain: str,
        trait: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Return the MLflow model URI that should be loaded for inference."""


class LocalArtifactResolver(ArtifactResolver):
    """Legacy resolver retained for compatibility; local artifact loading is disabled."""

    def resolve_model_path(
        self,
        domain: str,
        trait: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Resolve model path."""
        raise ValueError(
            "Local artifact resolution is disabled. "
            "Use MLflow registry URIs (models:/...) for inference."
        )


class RemoteArtifactResolver(ArtifactResolver):
    """Resolve inference model URIs from MLflow registry keys."""

    @staticmethod
    def _resolve_model_key(domain: str, trait: Optional[str]) -> str:
        """Resolve model key."""
        normalized_domain = str(domain or "").strip().lower()
        normalized_trait = str(trait or "").strip().lower()

        if normalized_domain == "personality":
            if not normalized_trait:
                raise ValueError("Personality model resolution requires trait.")
            return f"big5_{normalized_trait}_state"

        scope = f"{domain}:{trait}" if trait else str(domain)
        raise ValueError(f"Unsupported inference model scope '{scope}'.")

    def resolve_model_path(
        self,
        domain: str,
        trait: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Resolve model path."""
        model_key = self._resolve_model_key(domain=domain, trait=trait)
        return resolve_model_uri(
            model_key=model_key,
            model_uri=model_path,
        )


class FallbackArtifactResolver(ArtifactResolver):
    """Attempt a primary resolver first; local fallback is disabled."""

    def __init__(
        self,
        primary: Optional[ArtifactResolver] = None,
        fallback: Optional[ArtifactResolver] = None,
    ):
        """Initialize the fallback artifact resolver."""
        self.primary = primary or RemoteArtifactResolver()
        self.fallback = fallback or LocalArtifactResolver()

    def resolve_model_path(
        self,
        domain: str,
        trait: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Resolve model path."""
        try:
            return self.primary.resolve_model_path(
                domain=domain,
                trait=trait,
                model_path=model_path,
            )
        except NotImplementedError:
            logger.info(
                "Primary artifact resolver is not implemented for %s:%s. "
                "Attempting fallback resolver.",
                domain,
                trait or "default",
            )
        except Exception as exc:
            logger.warning(
                "Primary artifact resolution failed for %s:%s: %s. "
                "Attempting fallback resolver.",
                domain,
                trait or "default",
                exc,
            )

        return self.fallback.resolve_model_path(
            domain=domain,
            trait=trait,
            model_path=model_path,
        )
