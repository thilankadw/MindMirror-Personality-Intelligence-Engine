"""Dependency helpers."""
# app/db/deps.py
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database."""
    if AsyncSessionLocal is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured. Set DATABASE_URL to use this endpoint.",
        )
    async with AsyncSessionLocal() as session:
        try:
            await session.connection()
        except (TimeoutError, OSError, SQLAlchemyError) as exc:
            logger.exception("database connection failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed. Check DATABASE_URL and Supabase pooler availability.",
            ) from exc
        yield session
