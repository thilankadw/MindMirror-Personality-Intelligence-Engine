"""Extraction helpers for service."""
from datetime import datetime, timezone
from typing import Dict, Iterable

from src.extractors.base import ExtractionError, SocialPostExtractor
from src.extractors.models import ExtractedPosts, ExtractionRequest, StoredExtractionResult
from src.extractors.registry import InMemorySocialIdentityRegistry
from src.extractors.storage import JsonFileExtractionStorage


class SocialExtractionService:
    """Coordinate identity resolution, provider extraction, and output storage."""

    def __init__(
        self,
        extractors: Iterable[SocialPostExtractor],
        identity_registry: InMemorySocialIdentityRegistry,
        storage: JsonFileExtractionStorage,
    ):
        """Initialize the social extraction service."""
        self._extractors: Dict[str, SocialPostExtractor] = {}
        for extractor in extractors:
            self._extractors[extractor.provider] = extractor
        self._identity_registry = identity_registry
        self._storage = storage

    def extract_posts(self, request: ExtractionRequest) -> StoredExtractionResult:
        """Extract posts."""
        extractor = self._extractors.get(request.provider)
        if extractor is None:
            raise ExtractionError(
                f"No extractor registered for provider '{request.provider}'."
            )

        account = self._identity_registry.resolve_account(
            platform_username=request.platform_username,
            provider=request.provider,
            external_username=request.external_username,
        )
        posts = extractor.extract_posts(account)
        extracted = ExtractedPosts(
            platform_username=request.platform_username,
            provider=request.provider,
            external_username=account.external_username,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            posts=posts,
            metadata=dict(request.metadata),
        )
        return self._storage.save_posts(extracted, output_path=request.output_path)
