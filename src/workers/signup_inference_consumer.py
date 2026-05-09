"""Database-backed signup inference worker for personality jobs."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import UUID


ROOT_DIR = Path(__file__).resolve().parents[2]
API_DIR = ROOT_DIR / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.repositories.job_repository import (  # noqa: E402
    get_inference_job,
    get_next_queued_signup_job,
    mark_inference_job_done,
    mark_inference_job_failed,
    mark_inference_job_running,
)
from app.repositories.user_repository import get_platform_identity  # noqa: E402
from src.services.personality_signup_inference import run_personality_signup  # noqa: E402
from src.services.reddit_extractor import extract_reddit_user_bundle  # noqa: E402


logger = logging.getLogger(__name__)

SUPPORTED_SIGNUP_INFERENCE_DOMAINS = {"personality"}
STATIC_REDDIT_USERNAME = "ok_celery_4705"
STATIC_REDDIT_BUNDLE_PATH = ROOT_DIR / "src" / "data" / "data.json"


async def process_signup_inference_message(payload: dict) -> None:
    """Process one queued signup inference payload."""
    job_id = UUID(str(payload["job_id"]))
    user_id = UUID(str(payload["user_id"]))
    reddit_username = str(payload["reddit_username"])
    domains = [str(domain) for domain in payload.get("domains", [])]

    if AsyncSessionLocal is None:
        raise RuntimeError("Database is not configured for signup inference worker")

    normalized_domains = domains or ["personality"]
    unsupported_domains = [
        domain for domain in normalized_domains if domain not in SUPPORTED_SIGNUP_INFERENCE_DOMAINS
    ]
    if unsupported_domains:
        raise ValueError(f"Unsupported signup inference domain '{unsupported_domains[0]}'")

    async with AsyncSessionLocal() as session:
        job = await get_inference_job(session, job_id)
        if job is None:
            logger.warning("signup_inference.worker.job_missing job_id=%s", job_id)
            return

        await mark_inference_job_running(session, job)
        try:
            reddit_bundle = _load_reddit_bundle_for_signup(
                reddit_username=reddit_username,
                job_id=job_id,
            )
            await run_personality_signup(
                session,
                user_id=user_id,
                reddit_username=reddit_username,
                job_id=job_id,
                reddit_bundle=reddit_bundle,
            )
            await mark_inference_job_done(session, job)
        except Exception as exc:
            await session.rollback()
            logger.exception("signup_inference.worker.failed job_id=%s", job_id)
            failed_job = await get_inference_job(session, job_id)
            if failed_job is not None:
                await mark_inference_job_failed(session, failed_job, error=_safe_error_message(exc))
            return


async def consume_jobs_forever(poll_interval_seconds: float = 2.0) -> None:
    """Poll queued signup jobs from the database forever."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database is not configured for signup inference worker")

    logger.info(
        "signup_inference.worker.started poll_interval_seconds=%s",
        poll_interval_seconds,
    )
    while True:
        try:
            payload = await _next_signup_job_payload()
            if payload is None:
                await asyncio.sleep(poll_interval_seconds)
                continue
            await process_signup_inference_message(payload)
        except Exception:
            logger.exception("signup_inference.worker.loop_failed")
            await asyncio.sleep(poll_interval_seconds)


async def _next_signup_job_payload() -> dict | None:
    """Return the next queued signup payload if one exists."""
    async with AsyncSessionLocal() as session:
        job = await get_next_queued_signup_job(session)
        if job is None:
            return None

        identity = await get_platform_identity(session, user_id=job.user_id, platform="reddit")
        if identity is None:
            await mark_inference_job_failed(
                session,
                job,
                error="Reddit username is not linked for signup inference",
            )
            logger.warning(
                "signup_inference.worker.identity_missing job_id=%s user_id=%s",
                job.id,
                job.user_id,
            )
            return None

        return {
            "job_id": str(job.id),
            "user_id": str(job.user_id),
            "reddit_username": identity.platform_username,
            "domains": job.domains or ["personality"],
        }


def _load_reddit_bundle_for_signup(*, reddit_username: str, job_id: UUID) -> dict:
    """Load either a static demo bundle or a live Reddit bundle."""
    normalized_username = reddit_username.strip().lower()
    if normalized_username == STATIC_REDDIT_USERNAME:
        return _load_static_reddit_bundle(reddit_username=normalized_username, job_id=job_id)

    logger.info(
        "signup_inference.worker.reddit_extract.start job_id=%s reddit_username=%s",
        job_id,
        reddit_username,
    )
    reddit_bundle = extract_reddit_user_bundle(reddit_username)
    logger.info(
        "signup_inference.worker.reddit_extract.done job_id=%s posts=%s",
        job_id,
        len(reddit_bundle.get("posts") or []),
    )
    return reddit_bundle


def _load_static_reddit_bundle(*, reddit_username: str, job_id: UUID) -> dict:
    """Load the committed static demo Reddit bundle."""
    if not STATIC_REDDIT_BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Static Reddit bundle not found: {STATIC_REDDIT_BUNDLE_PATH}")

    logger.info(
        "signup_inference.worker.static_bundle.load job_id=%s reddit_username=%s path=%s",
        job_id,
        reddit_username,
        STATIC_REDDIT_BUNDLE_PATH,
    )
    with STATIC_REDDIT_BUNDLE_PATH.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    payload.setdefault("profile", {"username": reddit_username})
    payload.setdefault("posts", [])
    return payload


def _safe_error_message(exc: Exception, limit: int = 500) -> str:
    """Return a bounded string error message for job persistence."""
    message = str(exc).strip() or exc.__class__.__name__
    return message[:limit]


def main() -> None:
    """Run the database-backed worker."""
    logging.basicConfig(level="INFO")
    asyncio.run(consume_jobs_forever())


if __name__ == "__main__":
    main()
