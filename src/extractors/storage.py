"""Extraction helpers for storage."""
import json
import os
from typing import Any, Dict

from src.extractors.models import ExtractedPosts, StoredExtractionResult


class JsonFileExtractionStorage:
    """Persist extracted posts as normalized JSON documents."""

    def __init__(self, base_dir: str = "artifacts/extractions"):
        """Initialize the JSON file extraction storage."""
        self.base_dir = base_dir

    def save_posts(
        self,
        extracted: ExtractedPosts,
        output_path: str | None = None,
    ) -> StoredExtractionResult:
        """Save posts."""
        destination = output_path or self._default_output_path(
            platform_username=extracted.platform_username,
            provider=extracted.provider,
        )

        output_dir = os.path.dirname(destination)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload: Dict[str, Any] = {
            "platform_username": extracted.platform_username,
            "provider": extracted.provider,
            "external_username": extracted.external_username,
            "extracted_at": extracted.extracted_at,
            "posts_count": len(extracted.posts),
            "metadata": extracted.metadata,
            "posts": extracted.posts,
        }

        with open(destination, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)

        return StoredExtractionResult(
            platform_username=extracted.platform_username,
            provider=extracted.provider,
            external_username=extracted.external_username,
            extracted_at=extracted.extracted_at,
            output_path=destination,
            posts_count=len(extracted.posts),
        )

    def _default_output_path(self, platform_username: str, provider: str) -> str:
        """Handle default output path."""
        return os.path.join(self.base_dir, platform_username, provider, "posts.json")
