"""Base classes and shared interfaces."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.extractors.models import SocialAccountLink


class ExtractionError(RuntimeError):
    """Raised when an extractor cannot complete a provider request."""


class SocialPostExtractor(ABC):
    """Abstract extractor for a social platform's post stream."""

    provider: str

    @abstractmethod
    def extract_posts(self, account: SocialAccountLink) -> List[Dict[str, Any]]:
        """Fetch and normalize posts for a provider account."""
