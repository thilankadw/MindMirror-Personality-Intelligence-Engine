"""Run personality signup inference directly and persist the resulting payload."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DOMAINS = ["personality"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run personality signup inference directly and save predictions JSON in artifacts/."
    )
    parser.add_argument("--email", required=True, help="Email for local user context")
    parser.add_argument("--password", required=True, help="Password for local user context")
    parser.add_argument("--reddit-username", required=True, help="Reddit username for signup inference")
    parser.add_argument(
        "--sqlite-path",
        default="artifacts/tmp/signup_flow.db",
        help="SQLite DB path for this run (default: artifacts/tmp/signup_flow.db)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: artifacts/signup_flow/<reddit_username>_<timestamp>.json",
    )
    return parser.parse_args()


def _default_output_path(reddit_username: str) -> Path:
    """Build the default JSON output path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "signup_flow" / f"{reddit_username}_{timestamp}.json"


def _configure_environment(sqlite_path: Path) -> None:
    """Point the local run at a temporary SQLite database."""
    sqlite_path = sqlite_path.resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{sqlite_path.as_posix()}"
    os.environ["CREATE_TABLES_ON_STARTUP"] = "true"


def _configure_import_path(repo_root: Path) -> None:
    """Make the repository root and API package importable."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    api_dir = repo_root / "api"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))


def _json_safe(value: Any) -> Any:
    """Convert runtime values into JSON-safe data."""
    return json.loads(json.dumps(value, default=str))


async def _run(args: argparse.Namespace, repo_root: Path) -> Path:
    """Execute the direct personality signup inference flow."""
    _configure_environment(Path(args.sqlite_path))
    _configure_import_path(repo_root)

    from app.db.base import Base
    from app.db.models import (  # noqa: F401
        Comment,
        InferenceJob,
        Media,
        PersonalityInferenceRun,
        PersonalityWeeklyPrediction,
        Post,
        User,
        UserInferenceSnapshot,
        UserPlatformIdentity,
    )
    from app.db.session import AsyncSessionLocal, engine
    from app.repositories.job_repository import (
        create_inference_job,
        mark_inference_job_done,
        mark_inference_job_failed,
        mark_inference_job_running,
    )
    from app.repositories.prediction_repository import (
        format_personality_predictions_payload,
        get_personality_predictions_for_run,
        get_personality_run_for_job,
    )
    from app.repositories.user_repository import upsert_user_platform_identity
    from app.schemas.user import UserCreate
    from app.services.auth_service import create_user, get_user_by_email
    from app.services.job_service import get_job_status_payload
    from src.services.personality_signup_inference import run_personality_signup

    if engine is None or AsyncSessionLocal is None:
        raise RuntimeError("Database engine/session is not configured")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    started_at = datetime.now(timezone.utc).isoformat()

    async with AsyncSessionLocal() as session:
        existing_user = await get_user_by_email(session, args.email)
        if existing_user is None:
            user = await create_user(
                session,
                UserCreate(
                    email=args.email,
                    password=args.password,
                ),
            )
        else:
            user = existing_user

        await upsert_user_platform_identity(
            session,
            user_id=user.id,
            platform="reddit",
            platform_username=args.reddit_username,
        )

        job = await create_inference_job(
            session,
            user_id=user.id,
            job_type="signup",
            domains=EXPECTED_DOMAINS,
        )
        await mark_inference_job_running(session, job)

        try:
            await run_personality_signup(
                session,
                user_id=user.id,
                reddit_username=args.reddit_username,
                job_id=job.id,
            )
            await mark_inference_job_done(session, job)
        except Exception as exc:  # pragma: no cover - runtime path
            await mark_inference_job_failed(session, job, error=str(exc)[:500])

        job_payload = await get_job_status_payload(session, user_id=user.id, job_id=job.id)

        personality_payload: dict[str, Any] | None = None
        run = await get_personality_run_for_job(session, job_id=job.id, user_id=user.id)
        if run is not None:
            predictions = await get_personality_predictions_for_run(session, run_id=run.id)
            personality_payload = _json_safe(
                format_personality_predictions_payload(
                    job_id=job.id,
                    reddit_username=run.reddit_username,
                    computed_at=run.computed_at,
                    predictions=predictions,
                )
            )

        output_payload = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "started_at": started_at,
                "expected_domains": EXPECTED_DOMAINS,
            },
            "user": {
                "email": args.email,
                "reddit_username": args.reddit_username,
            },
            "signup_result": _json_safe(job_payload),
            "personality": personality_payload,
        }

    output_path = Path(args.output) if args.output else _default_output_path(args.reddit_username)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, default=str)

    return output_path


def main() -> int:
    """Run the direct signup inference export."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = asyncio.run(_run(args, repo_root))
    print(f"Saved signup flow predictions JSON: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
