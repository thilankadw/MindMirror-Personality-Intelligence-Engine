"""Extraction helpers for registry."""
from typing import Dict, Optional

from src.extractors.base import ExtractionError
from src.extractors.models import PlatformUserIdentity, SocialAccountLink


class InMemorySocialIdentityRegistry:
    """Resolve provider-specific usernames from the platform username."""

    def __init__(self, identities: Optional[Dict[str, PlatformUserIdentity]] = None):
        """Initialize the in memory social identity registry."""
        self._identities = dict(identities or {})

    def register(self, identity: PlatformUserIdentity) -> None:
        """Register the requested data."""
        self._identities[identity.platform_username] = identity

    def resolve_account(
        self,
        platform_username: str,
        provider: str,
        external_username: Optional[str] = None,
    ) -> SocialAccountLink:
        """Resolve account."""
        if external_username:
            return SocialAccountLink(provider=provider, external_username=external_username)

        identity = self._identities.get(platform_username)
        if identity is None:
            raise ExtractionError(
                f"No social account mapping found for platform user '{platform_username}'."
            )

        for account in identity.social_accounts:
            if account.provider == provider:
                return account

        raise ExtractionError(
            f"Platform user '{platform_username}' has no mapped account for provider '{provider}'."
        )
