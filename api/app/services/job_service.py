"""Services for job service."""
from __future__ import annotations

import asyncio
from time import monotonic
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.job_repository import get_inference_job_for_user
from app.repositories.prediction_repository import (
    format_personality_predictions_payload,
    get_personality_predictions_for_run,
    get_personality_run_for_job,
)


def build_job_result_url(job_id: UUID) -> str:
    """Build job result URL."""
    return f"{settings.API_V1_PREFIX}/inference/jobs/{job_id}"


def _normalize_domains(domains: object) -> list[str]:
    """Normalize domains."""
    if not isinstance(domains, list):
        return ["personality"]
    normalized = [str(domain) for domain in domains if str(domain).strip()]
    return normalized or ["personality"]


async def get_job_status_payload(
    db: AsyncSession,
    *,
    user_id: UUID,
    job_id: UUID,
) -> dict:
    """Get job status payload."""
    job = await get_inference_job_for_user(db, job_id=job_id, user_id=user_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inference job not found")

    if job.status != "done":
        payload = {
            "job_id": job.id,
            "status": job.status,
            "domains": _normalize_domains(job.domains),
        }
        if job.status in {"queued", "running"}:
            payload["result_url"] = build_job_result_url(job.id)
        if job.status == "failed":
            payload["error"] = "Unable to compute signup inference right now"
        return payload

    run = await get_personality_run_for_job(db, job_id=job.id, user_id=user_id)
    if run is None:
        return {
            "job_id": job.id,
            "status": "failed",
            "error": "Inference job completed without persisted results",
            "domains": _normalize_domains(job.domains),
        }

    predictions = await get_personality_predictions_for_run(db, run_id=run.id)
    payload = format_personality_predictions_payload(
        job_id=job.id,
        reddit_username=run.reddit_username,
        computed_at=run.computed_at,
        predictions=predictions,
    )
    payload["domains"] = _normalize_domains(job.domains)
    return payload


async def wait_for_job_status_payload(
    db: AsyncSession,
    *,
    user_id: UUID,
    job_id: UUID,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.3,
) -> tuple[bool, dict]:
    """Handle wait for job status payload."""
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        payload = await get_job_status_payload(db, user_id=user_id, job_id=job_id)
        status_value = payload.get("status")
        if status_value in {"done", "failed"} or "domain" in payload:
            return True, payload
        await asyncio.sleep(poll_interval_seconds)

    payload = await get_job_status_payload(db, user_id=user_id, job_id=job_id)
    status_value = payload.get("status")
    return status_value in {"done", "failed"} or "domain" in payload, payload
