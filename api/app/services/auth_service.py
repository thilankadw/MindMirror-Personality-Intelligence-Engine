"""Services for auth service."""
# app/services/auth_service.py
from __future__ import annotations

import re
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.schemas.user import UserCreate
from app.utils.security import create_access_token, hash_password, verify_password


logger = logging.getLogger(__name__)


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Get user by email."""
    logger.info("auth.db.lookup_user_by_email.start email=%s", email.lower())
    result = await session.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    logger.info("auth.db.lookup_user_by_email.done email=%s found=%s", email.lower(), user is not None)
    return user


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    """Get user by username."""
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def build_unique_username_from_email(session: AsyncSession, email: str) -> str:
    """Build unique username from email."""
    local_part = email.split("@", 1)[0].lower()
    base = re.sub(r"[^a-z0-9_]", "_", local_part).strip("_") or "user"
    candidate = base
    suffix = 1
    while await get_user_by_username(session, candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


async def create_user(session: AsyncSession, user_in: UserCreate) -> User:
    """Create user."""
    normalized_email = user_in.email.lower()
    reddit_username = user_in.reddit_username.strip().lower() if user_in.reddit_username else None
    logger.info("auth.signup.db.create_user.start email=%s reddit_username=%s", normalized_email, reddit_username)
    user = User(
        email=normalized_email,
        hashed_password=hash_password(user_in.password),
        username=await build_unique_username_from_email(session, normalized_email),
        reddit_username=reddit_username,
        auth_provider="local",
    )
    session.add(user)
    logger.info("auth.signup.db.commit_user.start email=%s", normalized_email)
    await session.commit()
    await session.refresh(user)
    logger.info("auth.signup.db.create_user.done user_id=%s username=%s", user.id, user.username)
    return user


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User | None:
    """Authenticate user."""
    user = await get_user_by_email(session, email)
    if not user or not user.is_active or not verify_password(password, user.hashed_password):
        return None
    return user


def issue_access_token(subject: str) -> str:
    """Handle issue access token."""
    return create_access_token(subject)
