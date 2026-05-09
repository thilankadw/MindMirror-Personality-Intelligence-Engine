"""Schemas for user reddit."""
from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.job import PersonalityJobResultOut


REDDIT_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


class RedditUsernameIn(BaseModel):
    """Schema for reddit username in."""
    reddit_username: str = Field(min_length=3, max_length=20)

    @field_validator("reddit_username")
    @classmethod
    def validate_reddit_username(cls, value: str) -> str:
        """Validate reddit username."""
        normalized = value.strip().lower()
        if " " in normalized:
            raise ValueError("Reddit username cannot contain spaces")
        if not REDDIT_USERNAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Reddit username must be 3-20 characters and use only letters, numbers, underscores, or hyphens"
            )
        return normalized


class SignupInferenceOut(PersonalityJobResultOut):
    """Schema for signup inference out."""
    pass


class UserMeOut(BaseModel):
    """Schema for user me out."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    is_active: bool
    reddit_username: str | None = None
    last_signup_computed_at: datetime | None = None
