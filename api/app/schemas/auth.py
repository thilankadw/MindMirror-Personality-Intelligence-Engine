"""Schemas for auth."""
# app/schemas/auth.py
from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

class Token(BaseModel):
    """Schema for token."""
    access_token: str
    token_type: str = "bearer"
    reddit_username: Optional[str] = None


class RedditOAuthAuthorizeIn(BaseModel):
    """Schema for reddit o auth authorize in."""
    redirect_uri: str = Field(min_length=1, max_length=2048)
    intent: Literal["signup", "signin"] = "signin"

    @field_validator("redirect_uri")
    @classmethod
    def validate_redirect_uri(cls, value: str) -> str:
        """Validate redirect URI."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("redirect_uri is required")
        return normalized


class RedditOAuthAuthorizeOut(BaseModel):
    """Schema for reddit o auth authorize out."""
    authorization_url: str
    state: str
    expires_in_seconds: int


class RedditOAuthExchangeIn(BaseModel):
    """Schema for reddit o auth exchange in."""
    code: str = Field(min_length=1, max_length=4096)
    state: str = Field(min_length=1, max_length=4096)
    redirect_uri: str = Field(min_length=1, max_length=2048)

    @field_validator("code", "state", "redirect_uri")
    @classmethod
    def strip_required_fields(cls, value: str) -> str:
        """Handle strip required fields."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value is required")
        return normalized


class RedditOAuthToken(Token):
    """Schema for reddit o auth token."""
    reddit_username: str
    is_new_user: bool


class TokenData(BaseModel):
    """Schema for token data."""
    username: Optional[str] = None


class TokenPayload(BaseModel):
    """Schema for token payload."""
    sub: str


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
