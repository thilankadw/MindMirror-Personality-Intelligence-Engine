"""Idempotent schema bootstrap for fresh databases (e.g. Supabase)."""

from __future__ import annotations

import asyncio

from sqlalchemy import inspect

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import engine


async def ensure_schema() -> None:
    """Ensure schema."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        def _table_names(sync_conn):
            """Return the sorted public table names for the active connection."""
            return sorted(inspect(sync_conn).get_table_names(schema="public"))

        table_names = await conn.run_sync(_table_names)

    print(f"Schema bootstrap complete. public tables: {len(table_names)}")
    for table_name in table_names:
        print(f"- {table_name}")


if __name__ == "__main__":
    asyncio.run(ensure_schema())
