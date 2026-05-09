"""Schemas for user."""
# app/schemas/user.py
from __future__ import annotations

from datetime import datetime
from uuid import UUID
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

class UserCreate(BaseModel):
    """Schema for user create."""
    username: str

class UserResponse(BaseModel):
    """Schema for user response."""
    id: uuid.UUID
    username: str
    pipeline_step: str
    pipeline_progress: float
    pipeline_message: Optional[str] = None

    class Config:
        """Schema for config."""
        from_attributes = True

class UserBase(BaseModel):
    """Schema for user base."""
    email: EmailStr


class UserCreate(UserBase):
    """Schema for user create."""
    password: str = Field(min_length=8, max_length=128)
    reddit_username: str | None = Field(default=None, min_length=3, max_length=20)

    @field_validator("reddit_username")
    @classmethod
    def normalize_reddit_username(cls, value: str | None) -> str | None:
        """Normalize reddit username."""
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None


class UserPublic(UserBase):
    """Schema for user public."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    is_active: bool
    created_at: datetime
