"""Dependency helpers."""
from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.deps import get_db
from app.db.models.user import User
from app.repositories.user_repository import get_user_by_id
from app.schemas.auth import TokenPayload


oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/signin")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db),
) -> User:
    """Get current user."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        token_data = TokenPayload(**payload)
    except JWTError:
        raise credentials_error

    user = None
    try:
        user_id = uuid.UUID(token_data.sub)
        user = await get_user_by_id(session, user_id)
    except ValueError:
        result = await session.execute(select(User).where(User.username == token_data.sub))
        user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_error
    return user
