"""Repository helpers for user repository."""
from __future__ import annotations

from datetime import datetime, timezone
import re
import secrets
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.db.models.user import User
from app.db.models.prediction import PersonalityInferenceRun
from app.db.models.user_inference_snapshot import UserInferenceSnapshot
from app.db.models.user_platform_identity import UserPlatformIdentity
from app.utils.security import hash_password


async def get_user_by_id(session: AsyncSession, user_id: UUID) -> User | None:
    """Get user by ID."""
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_platform_identity(
    session: AsyncSession,
    user_id: UUID,
    platform: str,
) -> UserPlatformIdentity | None:
    """Get platform identity."""
    result = await session.execute(
        select(UserPlatformIdentity).where(
            UserPlatformIdentity.user_id == user_id,
            UserPlatformIdentity.platform == platform,
        )
    )
    return result.scalar_one_or_none()


async def get_platform_identity_by_username(
    session: AsyncSession,
    platform: str,
    platform_username: str,
) -> UserPlatformIdentity | None:
    """Get platform identity by username."""
    result = await session.execute(
        select(UserPlatformIdentity).where(
            UserPlatformIdentity.platform == platform,
            UserPlatformIdentity.platform_username == platform_username,
        )
    )
    return result.scalar_one_or_none()


async def upsert_user_platform_identity(
    session: AsyncSession,
    user_id: UUID,
    platform: str,
    platform_username: str,
) -> UserPlatformIdentity:
    """Upsert user platform identity."""
    existing_for_username = await get_platform_identity_by_username(
        session,
        platform=platform,
        platform_username=platform_username,
    )
    if existing_for_username and existing_for_username.user_id != user_id:
        raise ValueError(f"{platform.title()} username is already linked to another account")

    identity = await get_platform_identity(session, user_id=user_id, platform=platform)
    if identity is None:
        identity = UserPlatformIdentity(
            user_id=user_id,
            platform=platform,
            platform_username=platform_username,
        )
        session.add(identity)
    else:
        identity.platform_username = platform_username

    await session.commit()
    await session.refresh(identity)
    return identity


async def get_latest_snapshot_by_kind(
    session: AsyncSession,
    user_id: UUID,
    kind: str,
) -> UserInferenceSnapshot | None:
    """Get latest snapshot by kind."""
    result = await session.execute(
        select(UserInferenceSnapshot).where(
            UserInferenceSnapshot.user_id == user_id,
            UserInferenceSnapshot.kind == kind,
        )
    )
    return result.scalar_one_or_none()


async def get_latest_personality_run(session: AsyncSession, user_id: UUID) -> PersonalityInferenceRun | None:
    """Get latest personality run."""
    result = await session.execute(
        select(PersonalityInferenceRun)
        .where(PersonalityInferenceRun.user_id == user_id)
        .order_by(PersonalityInferenceRun.computed_at.desc())
    )
    return result.scalars().first()


async def upsert_user_inference_snapshot(
    session: AsyncSession,
    user_id: UUID,
    kind: str,
    payload_json: dict,
    model_versions_json: dict | None = None,
    computed_at: datetime | None = None,
) -> UserInferenceSnapshot:
    """Upsert user inference snapshot."""
    snapshot = await get_latest_snapshot_by_kind(session, user_id=user_id, kind=kind)
    computed_at = computed_at or datetime.now(timezone.utc)

    if snapshot is None:
        snapshot = UserInferenceSnapshot(
            user_id=user_id,
            kind=kind,
            payload_json=payload_json,
            computed_at=computed_at,
            model_versions_json=model_versions_json,
        )
        session.add(snapshot)
    else:
        snapshot.payload_json = payload_json
        snapshot.computed_at = computed_at
        snapshot.model_versions_json = model_versions_json

    await session.commit()
    await session.refresh(snapshot)
    return snapshot

class UserRepository:
    """Persist user records."""
    def __init__(self, session: AsyncSession):
        """Initialize the user repository."""
        self.session = session

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get by username."""
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_by_reddit_username(self, reddit_username: str) -> Optional[User]:
        """Get by reddit username."""
        result = await self.session.execute(select(User).where(User.reddit_username == reddit_username))
        return result.scalar_one_or_none()

    async def get_by_reddit_oauth_id(self, reddit_oauth_id: str) -> Optional[User]:
        """Get by reddit OAuth ID."""
        result = await self.session.execute(select(User).where(User.reddit_oauth_id == reddit_oauth_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get by email."""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def _build_legacy_email(self, username: str) -> str:
        """Build legacy email."""
        safe_username = re.sub(r"[^a-z0-9_]", "_", username.strip().lower())
        safe_username = safe_username.strip("_") or "legacy_user"
        candidate = f"{safe_username}@legacy.mindmirror.local"
        counter = 1
        while await self.get_by_email(candidate):
            candidate = f"{safe_username}_{counter}@legacy.mindmirror.local"
            counter += 1
        return candidate

    async def create(self, username: str, reddit_username: str = None) -> User:
        """Create the requested data."""
        normalized_username = username.strip().lower()
        user = User(
            email=await self._build_legacy_email(normalized_username),
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            username=normalized_username,
            reddit_username=(reddit_username or normalized_username).strip().lower(),
            auth_provider="legacy",
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
