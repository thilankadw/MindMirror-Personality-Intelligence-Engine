"""Database models for user."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    """Database model for user."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[str | None] = mapped_column(String(50), unique=True, index=True, nullable=True)
    reddit_username: Mapped[str | None] = mapped_column(String(50), unique=True, index=True, nullable=True)
    reddit_oauth_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    auth_provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="local",
        server_default=text("'local'"),
    )
    comment_karma: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    link_karma: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    total_karma: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    is_gold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    is_mod: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    personal_model_present: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    personal_data_present: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    personal_model_training: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    personal_model_training_counts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    pipeline_step: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="IDLE",
        server_default=text("'IDLE'"),
    )
    pipeline_progress: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default=text("0.0"),
    )
    pipeline_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reddit_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    reddit_access_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reddit_scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
