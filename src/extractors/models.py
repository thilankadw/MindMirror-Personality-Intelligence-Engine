"""Extraction helpers for models."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SocialAccountLink:
    """Represent social account link."""
    provider: str
    external_username: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlatformUserIdentity:
    """Represent platform user identity."""
    platform_username: str
    social_accounts: List[SocialAccountLink] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionRequest:
    """Represent extraction request."""
    platform_username: str
    provider: str
    external_username: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedPosts:
    """Represent extracted posts."""
    platform_username: str
    provider: str
    external_username: str
    extracted_at: str
    posts: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredExtractionResult:
    """Represent stored extraction result."""
    platform_username: str
    provider: str
    external_username: str
    extracted_at: str
    output_path: str
    posts_count: int
