"""Services for reddit OAuth service."""
from __future__ import annotations

import asyncio
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urlencode

import requests
from jose import JWTError, jwt
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.user import User
from app.repositories.user_repository import upsert_user_platform_identity
from app.utils.security import hash_password


class RedditOAuthError(RuntimeError):
    """Error raised for reddit o auth."""
    def __init__(self, message: str, *, status_code: int = 400):
        """Initialize the reddit o auth error."""
        super().__init__(message)
        self.status_code = status_code


class RedditOAuthConfigError(RedditOAuthError):
    """Error raised for reddit o auth config."""
    def __init__(self, message: str):
        """Initialize the reddit o auth config error."""
        super().__init__(message, status_code=503)


class RedditOAuthUpstreamError(RedditOAuthError):
    """Error raised for reddit o auth upstream."""
    def __init__(self, message: str):
        """Initialize the reddit o auth upstream error."""
        super().__init__(message, status_code=502)


class RedditOAuthConflictError(RedditOAuthError):
    """Error raised for reddit o auth conflict."""
    def __init__(self, message: str):
        """Initialize the reddit o auth conflict error."""
        super().__init__(message, status_code=409)


class RedditOAuthNotFoundError(RedditOAuthError):
    """Error raised for reddit o auth not found."""
    def __init__(self, message: str):
        """Initialize the reddit o auth not found error."""
        super().__init__(message, status_code=404)


@dataclass
class RedditOAuthAuthResult:
    """Represent reddit o auth auth result."""
    user: User
    reddit_username: str
    is_new_user: bool


def _require_reddit_oauth_config() -> None:
    """Handle require reddit OAuth config."""
    if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
        raise RedditOAuthConfigError(
            "Reddit OAuth is not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )


def _normalize_reddit_username(reddit_username: str) -> str:
    """Normalize reddit username."""
    normalized = reddit_username.strip().lower()
    if not normalized:
        raise RedditOAuthError("Reddit username is missing from provider response.")
    return normalized


def _normalize_username(username: str) -> str:
    """Normalize username."""
    base = re.sub(r"[^a-z0-9_]", "_", username.strip().lower())
    base = base.strip("_")
    return base or "reddit_user"


def _build_oauth_state(*, redirect_uri: str, intent: Literal["signup", "signin"]) -> str:
    """Build OAuth state."""
    now = datetime.now(timezone.utc)
    payload = {
        "typ": "reddit_oauth_state",
        "nonce": secrets.token_urlsafe(16),
        "intent": intent,
        "redirect_uri": redirect_uri,
        "iat": now,
        "exp": now + timedelta(seconds=settings.REDDIT_OAUTH_STATE_EXP_SECONDS),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def _validate_oauth_state(*, state: str, redirect_uri: str, intent: Literal["signup", "signin"]) -> None:
    """Validate OAuth state."""
    try:
        payload = jwt.decode(state, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise RedditOAuthError("Invalid or expired OAuth state.") from exc

    if payload.get("typ") != "reddit_oauth_state":
        raise RedditOAuthError("Invalid OAuth state type.")
    if payload.get("intent") != intent:
        raise RedditOAuthError("OAuth state does not match requested action.")
    if payload.get("redirect_uri") != redirect_uri:
        raise RedditOAuthError("OAuth state does not match redirect URI.")


def build_reddit_authorization_payload(
    *,
    redirect_uri: str,
    intent: Literal["signup", "signin"],
) -> dict:
    """Build reddit authorization payload."""
    _require_reddit_oauth_config()
    state = _build_oauth_state(redirect_uri=redirect_uri, intent=intent)
    params = {
        "client_id": settings.REDDIT_CLIENT_ID,
        "response_type": "code",
        "state": state,
        "redirect_uri": redirect_uri,
        "duration": settings.REDDIT_OAUTH_DURATION,
        "scope": settings.REDDIT_OAUTH_SCOPE,
    }
    return {
        "authorization_url": f"{settings.REDDIT_OAUTH_AUTHORIZE_URL}?{urlencode(params)}",
        "state": state,
        "expires_in_seconds": settings.REDDIT_OAUTH_STATE_EXP_SECONDS,
    }


def _exchange_code_for_token(*, code: str, redirect_uri: str) -> dict:
    """Exchange code for token."""
    response = requests.post(
        settings.REDDIT_OAUTH_TOKEN_URL,
        auth=(settings.REDDIT_CLIENT_ID, settings.REDDIT_CLIENT_SECRET),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        headers={
            "User-Agent": settings.REDDIT_USER_AGENT,
            "Accept": "application/json",
        },
        timeout=settings.REDDIT_OAUTH_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail = response.text[:300]
        raise RedditOAuthUpstreamError(
            f"Reddit token exchange failed (HTTP {response.status_code}): {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RedditOAuthUpstreamError("Reddit token response is not valid JSON.") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise RedditOAuthUpstreamError("Reddit token response did not include an access token.")
    return payload


def _fetch_reddit_identity(access_token: str) -> dict:
    """Fetch reddit identity."""
    response = requests.get(
        settings.REDDIT_OAUTH_ME_URL,
        headers={
            "Authorization": f"bearer {access_token}",
            "User-Agent": settings.REDDIT_USER_AGENT,
            "Accept": "application/json",
        },
        timeout=settings.REDDIT_OAUTH_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail = response.text[:300]
        raise RedditOAuthUpstreamError(
            f"Reddit identity fetch failed (HTTP {response.status_code}): {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RedditOAuthUpstreamError("Reddit identity response is not valid JSON.") from exc

    if not payload.get("name") or not payload.get("id"):
        raise RedditOAuthUpstreamError("Reddit identity payload is missing required fields.")
    return payload


async def _get_user_by_reddit_oauth_id(session: AsyncSession, reddit_oauth_id: str) -> User | None:
    """Get user by reddit OAuth ID."""
    result = await session.execute(select(User).where(User.reddit_oauth_id == reddit_oauth_id))
    return result.scalar_one_or_none()


async def _get_user_by_reddit_username(session: AsyncSession, reddit_username: str) -> User | None:
    """Get user by reddit username."""
    result = await session.execute(select(User).where(User.reddit_username == reddit_username))
    return result.scalar_one_or_none()


async def _get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Get user by email."""
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def _get_user_by_username(session: AsyncSession, username: str) -> User | None:
    """Get user by username."""
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def _build_unique_username(session: AsyncSession, desired_username: str) -> str:
    """Build unique username."""
    base = _normalize_username(desired_username)
    candidate = base
    suffix = 1
    while await _get_user_by_username(session, candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


async def _build_unique_oauth_email(
    session: AsyncSession,
    *,
    reddit_username: str,
    reddit_oauth_id: str,
) -> str:
    """Build unique OAuth email."""
    safe_username = re.sub(r"[^a-z0-9_]", "_", reddit_username.lower()).strip("_") or "reddit_user"
    safe_id = re.sub(r"[^a-z0-9_]", "_", reddit_oauth_id.lower()).strip("_") or "id"
    candidate = f"reddit_{safe_username}_{safe_id}@oauth.reddit.local"
    suffix = 1
    while await _get_user_by_email(session, candidate):
        candidate = f"reddit_{safe_username}_{safe_id}_{suffix}@oauth.reddit.local"
        suffix += 1
    return candidate


def _to_int(value: object) -> int:
    """Convert a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def complete_reddit_oauth(
    *,
    session: AsyncSession,
    code: str,
    state: str,
    redirect_uri: str,
    intent: Literal["signup", "signin"],
) -> RedditOAuthAuthResult:
    """Complete reddit OAuth."""
    _require_reddit_oauth_config()
    _validate_oauth_state(state=state, redirect_uri=redirect_uri, intent=intent)

    token_payload = await asyncio.to_thread(_exchange_code_for_token, code=code, redirect_uri=redirect_uri)
    identity_payload = await asyncio.to_thread(
        _fetch_reddit_identity, access_token=str(token_payload["access_token"])
    )

    reddit_oauth_id = str(identity_payload["id"]).strip()
    reddit_username = _normalize_reddit_username(str(identity_payload["name"]))

    user_by_oauth = await _get_user_by_reddit_oauth_id(session, reddit_oauth_id)
    user_by_username = await _get_user_by_reddit_username(session, reddit_username)
    if user_by_oauth and user_by_username and user_by_oauth.id != user_by_username.id:
        raise RedditOAuthConflictError("This Reddit account is already linked to a different user.")
    if (
        user_by_username
        and user_by_username.reddit_oauth_id
        and user_by_username.reddit_oauth_id != reddit_oauth_id
    ):
        raise RedditOAuthConflictError("This Reddit username is already linked to another account.")

    is_new_user = False
    user: User | None = user_by_oauth or user_by_username

    if intent == "signup" and user_by_oauth:
        raise RedditOAuthConflictError("This Reddit account is already registered.")
    if intent == "signin" and user is None:
        raise RedditOAuthNotFoundError("No account found for this Reddit user. Please sign up first.")

    if user is None:
        is_new_user = True
        user = User(
            email=await _build_unique_oauth_email(
                session, reddit_username=reddit_username, reddit_oauth_id=reddit_oauth_id
            ),
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            username=await _build_unique_username(session, reddit_username),
            reddit_username=reddit_username,
            reddit_oauth_id=reddit_oauth_id,
            auth_provider="reddit",
            verified=bool(identity_payload.get("verified", False)),
        )
        session.add(user)
        try:
            await session.commit()
            await session.refresh(user)
        except IntegrityError as exc:
            await session.rollback()
            raise RedditOAuthConflictError("Unable to create user because identity is already linked.") from exc

    user.reddit_username = reddit_username
    user.reddit_oauth_id = reddit_oauth_id
    user.comment_karma = _to_int(identity_payload.get("comment_karma"))
    user.link_karma = _to_int(identity_payload.get("link_karma"))
    user.total_karma = user.comment_karma + user.link_karma
    user.is_mod = bool(identity_payload.get("is_mod", False))
    user.is_gold = bool(identity_payload.get("is_gold", False))
    user.verified = bool(identity_payload.get("verified", user.verified))
    user.auth_provider = "reddit"
    user.reddit_refresh_token = token_payload.get("refresh_token")
    user.reddit_scope = token_payload.get("scope")
    expires_in = _to_int(token_payload.get("expires_in"))
    user.reddit_access_token_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        if expires_in > 0
        else None
    )

    try:
        await session.commit()
        await session.refresh(user)
    except IntegrityError as exc:
        await session.rollback()
        raise RedditOAuthConflictError("Unable to link Reddit account due to an identity conflict.") from exc

    try:
        await upsert_user_platform_identity(
            session,
            user_id=user.id,
            platform="reddit",
            platform_username=reddit_username,
        )
    except ValueError as exc:
        raise RedditOAuthConflictError(str(exc)) from exc

    return RedditOAuthAuthResult(
        user=user,
        reddit_username=reddit_username,
        is_new_user=is_new_user,
    )
