"""Database models for comment."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, Float, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Comment(Base):
    """Database model for comment."""
    __tablename__ = "comments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comment_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    author_username: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    link_id: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String(20))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_utc: Mapped[float] = mapped_column(Float, nullable=False)
    created_date: Mapped[Optional[str]] = mapped_column(String(20))
    permalink: Mapped[Optional[str]] = mapped_column(String(500))
    first_fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
