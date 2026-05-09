"""Schemas for job."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class JobResponse(BaseModel):
    """Schema for job response."""
    job_id: str
    status: str
    result: Optional[dict] = None

class TraitPredictionOut(BaseModel):
    """Schema for trait prediction out."""
    direction: str
    score: float | None = None


class WeeklyPredictionOut(BaseModel):
    """Schema for weekly prediction out."""
    week_start: date
    directions: dict[str, TraitPredictionOut]


class PersonalityJobResultOut(BaseModel):
    """Schema for personality job result out."""
    model_config = ConfigDict(extra="ignore")

    job_id: UUID
    reddit_username: str
    domain: Literal["personality"]
    predictions: list[WeeklyPredictionOut]
    computed_at: datetime
    domains: list[str] = Field(default_factory=list)


class InferenceJobQueuedOut(BaseModel):
    """Schema for inference job queued out."""
    job_id: UUID
    status: Literal["queued", "running"]
    result_url: str | None = None
    domains: list[str] = Field(default_factory=list)


class InferenceJobFailedOut(BaseModel):
    """Schema for inference job failed out."""
    job_id: UUID
    status: Literal["failed"]
    error: str
    domains: list[str] = Field(default_factory=list)
