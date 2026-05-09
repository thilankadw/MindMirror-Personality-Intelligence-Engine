"""Services for signup inference service."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.job_repository import create_inference_job
from app.repositories.user_repository import upsert_user_platform_identity
from app.services.job_service import wait_for_job_status_payload


logger = logging.getLogger(__name__)


class SignupQueueError(RuntimeError):
    """Error raised when signup inference cannot be queued."""
    def __init__(self, message: str, *, job_id: UUID):
        """Initialize the signup queue error."""
        super().__init__(message)
        self.job_id = job_id


async def queue_signup_inference_for_user(
    db: AsyncSession,
    user_id: UUID,
    reddit_username: str,
    wait_for_completion: bool = True,
) -> tuple[bool, dict]:
    """Queue signup inference for user."""
    logger.info(
        "signup_inference.queue.start user_id=%s reddit_username=%s wait_for_completion=%s",
        user_id,
        reddit_username,
        wait_for_completion,
    )
    logger.info("signup_inference.identity.upsert.start user_id=%s platform=reddit", user_id)
    await upsert_user_platform_identity(
        db,
        user_id=user_id,
        platform="reddit",
        platform_username=reddit_username,
    )
    logger.info("signup_inference.identity.upsert.done user_id=%s platform=reddit", user_id)
    logger.info("signup_inference.job.create.start user_id=%s job_type=signup", user_id)
    job = await create_inference_job(
        db,
        user_id=user_id,
        job_type="signup",
        domains=["personality"],
    )
    logger.info(
        "signup_inference.job.create.done job_id=%s user_id=%s domains=%s",
        job.id,
        user_id,
        job.domains,
    )
    logger.info("signup_inference.db_queue.ready job_id=%s", job.id)

    if not wait_for_completion:
        logger.info("signup_inference.queue.return_queued job_id=%s", job.id)
        return (
            False,
            {
                "job_id": job.id,
                "status": "queued",
                "result_url": f"{settings.API_V1_PREFIX}/inference/jobs/{job.id}",
                "domains": job.domains,
            },
        )

    logger.info(
        "signup_inference.wait_for_completion.start job_id=%s timeout_seconds=%s",
        job.id,
        settings.SIGNUP_SYNC_WAIT_SECONDS,
    )
    return await wait_for_job_status_payload(
        db,
        user_id=user_id,
        job_id=job.id,
        timeout_seconds=settings.SIGNUP_SYNC_WAIT_SECONDS,
    )


async def publish_signup_requested(
    *,
    job_id: UUID,
    user_id: UUID,
    reddit_username: str,
    domains: list[str],
) -> None:
    """Keep a compatibility hook for callers expecting an explicit queue step."""
    job_message = {
        "job_id": str(job_id),
        "user_id": str(user_id),
        "reddit_username": reddit_username,
        "domains": domains,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        logger.info("signup_inference.queue.publish.skipped job_id=%s payload=%s", job_id, job_message)
    except Exception as exc:
        logger.exception("failed to enqueue signup inference request job_id=%s", job_id)
        raise SignupQueueError("Unable to queue signup inference", job_id=job_id) from exc
