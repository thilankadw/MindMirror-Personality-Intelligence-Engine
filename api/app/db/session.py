"""Utilities for session."""
# app/db/session.py
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings


engine = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None

# Only create engine when DATABASE_URL is configured
if settings.DATABASE_URL:
    engine = create_async_engine(
        str(settings.DATABASE_URL),
        pool_pre_ping=True,
        connect_args={
            "timeout": settings.DB_CONNECT_TIMEOUT_SECONDS,
            # Supabase poolers do not reliably support connection-local
            # prepared statement caches across pooled backend sessions.
            "prepared_statement_cache_size": 0,
        },
    )
    AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
else:
    engine = None  # type: ignore[assignment]
    AsyncSessionLocal = None  # type: ignore[assignment]
