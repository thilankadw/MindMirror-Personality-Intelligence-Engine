"""Database models for post."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, Float, Text, Boolean, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Post(Base):
    """Database model for post."""
    __tablename__ = "posts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reddit_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    author_username: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    subreddit: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    selftext: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(500))
    permalink: Mapped[Optional[str]] = mapped_column(String(500))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    upvote_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    num_comments: Mapped[float] = mapped_column(Float, default=0.0)
    created_utc: Mapped[float] = mapped_column(Float, nullable=False)
    created_date: Mapped[Optional[str]] = mapped_column(String(20))

    is_self: Mapped[bool] = mapped_column(Boolean, default=False)
    is_video: Mapped[bool] = mapped_column(Boolean, default=False)
    is_nsfw: Mapped[bool] = mapped_column(Boolean, default=False)
    is_spoiler: Mapped[bool] = mapped_column(Boolean, default=False)
    stickied: Mapped[bool] = mapped_column(Boolean, default=False)
    is_original_content: Mapped[bool] = mapped_column(Boolean, default=False)

    first_fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    media: Mapped[list["Media"]] = relationship("Media", back_populates="post", cascade="all, delete-orphan")
