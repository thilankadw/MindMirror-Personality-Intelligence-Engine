"""Repository helpers for personality prediction persistence."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.prediction import PersonalityInferenceRun, PersonalityWeeklyPrediction


async def create_personality_inference_run(
    session: AsyncSession,
    job_id: UUID,
    user_id: UUID,
    reddit_username: str,
    run_type: str,
    computed_at: datetime,
    model_versions: dict | None = None,
) -> PersonalityInferenceRun:
    """Create a personality inference run."""
    run = PersonalityInferenceRun(
        job_id=job_id,
        user_id=user_id,
        reddit_username=reddit_username,
        run_type=run_type,
        computed_at=computed_at,
        model_versions=model_versions,
    )
    session.add(run)
    await session.flush()
    return run


async def add_personality_weekly_predictions(
    session: AsyncSession,
    *,
    run_id: UUID,
    user_id: UUID,
    predictions: list[dict],
) -> None:
    """Persist weekly personality predictions for a run."""
    rows = [
        PersonalityWeeklyPrediction(
            run_id=run_id,
            user_id=user_id,
            week_start=prediction["week_start"],
            trait=prediction["trait"],
            direction=prediction["direction"],
            score=prediction.get("score"),
        )
        for prediction in predictions
    ]
    session.add_all(rows)
    await session.flush()


async def get_personality_run_for_job(
    session: AsyncSession,
    *,
    job_id: UUID,
    user_id: UUID,
) -> PersonalityInferenceRun | None:
    """Return the latest personality inference run for a job."""
    result = await session.execute(
        select(PersonalityInferenceRun)
        .where(
            PersonalityInferenceRun.job_id == job_id,
            PersonalityInferenceRun.user_id == user_id,
        )
        .order_by(PersonalityInferenceRun.computed_at.desc())
    )
    return result.scalars().first()


async def get_personality_predictions_for_run(
    session: AsyncSession,
    *,
    run_id: UUID,
) -> list[PersonalityWeeklyPrediction]:
    """Return personality predictions ordered by week and trait."""
    result = await session.execute(
        select(PersonalityWeeklyPrediction)
        .where(PersonalityWeeklyPrediction.run_id == run_id)
        .order_by(PersonalityWeeklyPrediction.week_start.asc(), PersonalityWeeklyPrediction.trait.asc())
    )
    return list(result.scalars().all())


def format_personality_predictions_payload(
    *,
    job_id: UUID,
    reddit_username: str,
    computed_at: datetime,
    predictions: list[PersonalityWeeklyPrediction],
) -> dict:
    """Format persisted personality predictions for API responses."""
    grouped: dict[date, dict] = defaultdict(dict)
    for prediction in predictions:
        grouped[prediction.week_start][prediction.trait] = {
            "direction": prediction.direction,
            "score": prediction.score,
        }

    prediction_groups = [
        {
            "week_start": week_start,
            "directions": grouped[week_start],
        }
        for week_start in sorted(grouped.keys())
    ]

    return {
        "job_id": job_id,
        "reddit_username": reddit_username,
        "domain": "personality",
        "predictions": prediction_groups,
        "computed_at": computed_at,
    }
