"""Database models for user platform identity."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UserPlatformIdentity(Base):
    """Database model for user platform identity."""
    __tablename__ = "user_platform_identities"
    __table_args__ = (
        UniqueConstraint("user_id", "platform", name="uq_user_platform_identities_user_platform"),
        UniqueConstraint(
            "platform",
            "platform_username",
            name="uq_user_platform_identities_platform_username",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    platform: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="reddit",
        server_default=text("'reddit'"),
    )
    platform_username: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
