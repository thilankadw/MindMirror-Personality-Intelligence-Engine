"""Repository helpers for job repository."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.job import InferenceJob


async def create_inference_job(
    session: AsyncSession,
    user_id: UUID,
    job_type: str,
    domains: list[str],
) -> InferenceJob:
    """Create inference job."""
    job = InferenceJob(
        user_id=user_id,
        job_type=job_type,
        status="queued",
        domains=domains,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_inference_job(session: AsyncSession, job_id: UUID) -> InferenceJob | None:
    """Get inference job."""
    result = await session.execute(
        select(InferenceJob)
        .where(InferenceJob.id == job_id)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def get_next_queued_signup_job(session: AsyncSession) -> InferenceJob | None:
    """Get next queued signup job."""
    result = await session.execute(
        select(InferenceJob)
        .where(
            InferenceJob.job_type == "signup",
            InferenceJob.status == "queued",
        )
        .order_by(InferenceJob.created_at.asc())
        .limit(1)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def get_inference_job_for_user(
    session: AsyncSession,
    job_id: UUID,
    user_id: UUID,
) -> InferenceJob | None:
    """Get inference job for user."""
    result = await session.execute(
        select(InferenceJob)
        .where(
            InferenceJob.id == job_id,
            InferenceJob.user_id == user_id,
        )
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def mark_inference_job_running(session: AsyncSession, job: InferenceJob) -> InferenceJob:
    """Mark inference job running."""
    job.status = "running"
    job.started_at = datetime.now(timezone.utc)
    job.error = None
    await session.commit()
    await session.refresh(job)
    return job


async def mark_inference_job_done(session: AsyncSession, job: InferenceJob) -> InferenceJob:
    """Mark inference job done."""
    job.status = "done"
    job.finished_at = datetime.now(timezone.utc)
    job.error = None
    await session.commit()
    await session.refresh(job)
    return job


async def mark_inference_job_failed(
    session: AsyncSession,
    job: InferenceJob,
    error: str,
) -> InferenceJob:
    """Mark inference job failed."""
    job.status = "failed"
    job.finished_at = datetime.now(timezone.utc)
    job.error = error
    await session.commit()
    await session.refresh(job)
    return job

class JobRepository:
    """Persist job records."""
    def __init__(self, session):
        """Initialize the job repository."""
        self.session = session
        
    # Stub for future Job schemas
    pass
