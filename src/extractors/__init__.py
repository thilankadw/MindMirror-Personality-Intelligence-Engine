"""Social media extraction primitives and provider implementations."""

from src.extractors.base import ExtractionError, SocialPostExtractor
from src.extractors.models import (
    ExtractedPosts,
    ExtractionRequest,
    PlatformUserIdentity,
    SocialAccountLink,
    StoredExtractionResult,
)
from src.extractors.reddit import RedditPostExtractor
from src.extractors.registry import InMemorySocialIdentityRegistry
from src.extractors.service import SocialExtractionService
from src.extractors.storage import JsonFileExtractionStorage

__all__ = [
    "ExtractedPosts",
    "ExtractionError",
    "ExtractionRequest",
    "InMemorySocialIdentityRegistry",
    "JsonFileExtractionStorage",
    "PlatformUserIdentity",
    "RedditPostExtractor",
    "SocialAccountLink",
    "SocialExtractionService",
    "SocialPostExtractor",
    "StoredExtractionResult",
]
