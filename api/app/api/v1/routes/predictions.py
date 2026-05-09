"""API routes for predictions."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.deps import get_db
from app.db.models.user import User
from app.repositories.prediction_repository import (
    format_personality_predictions_payload,
    get_personality_predictions_for_run,
)
from app.repositories.user_repository import get_latest_personality_run


router = APIRouter(prefix="/users/me/predictions", tags=["predictions"])

async def _get_personality_payload(session: AsyncSession, current_user: User) -> dict[str, Any] | None:
    """Get personality payload."""
    latest_run = await get_latest_personality_run(session, current_user.id)
    if latest_run is None:
        return None

    predictions = await get_personality_predictions_for_run(session, run_id=latest_run.id)
    payload = format_personality_predictions_payload(
        job_id=latest_run.job_id,
        reddit_username=latest_run.reddit_username,
        computed_at=latest_run.computed_at,
        predictions=predictions,
    )
    payload["run_id"] = latest_run.id
    payload["model_versions"] = latest_run.model_versions or {}
    return payload


@router.get("/personality")
async def get_personality_predictions(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get personality predictions."""
    payload = await _get_personality_payload(session, current_user)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No personality predictions found for this user.",
        )
    return payload


@router.get("")
async def get_all_domain_predictions(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get all available prediction payloads."""
    personality = await _get_personality_payload(session, current_user)
    missing_domains = [] if personality is not None else ["personality"]

    return {
        "domains": ["personality"],
        "personality": personality,
        "missing_domains": missing_domains,
    }
