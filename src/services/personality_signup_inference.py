"""Services for personality signup inference."""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.repositories.prediction_repository import (
    add_personality_weekly_predictions,
    create_personality_inference_run,
)
from utils.config import PERSONALITY_TRAITS

from src.services.reddit_extractor import extract_reddit_user_bundle


DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
logger = logging.getLogger(__name__)


async def run_personality_signup(
    db,
    user_id: UUID,
    reddit_username: str,
    job_id: UUID,
    *,
    reddit_bundle: dict[str, Any] | None = None,
) -> None:
    """Run personality signup."""
    import pandas as pd

    from src.personality.inference_engine import PersonalityInferenceEngine
    from src.personality.inference_feature_builder import PersonalityInferenceFeatureBuilder

    logger.info(
        "signup_inference.personality.start user_id=%s job_id=%s reddit_username=%s",
        user_id,
        job_id,
        reddit_username,
    )
    bundle = (
        reddit_bundle
        if reddit_bundle is not None
        else extract_reddit_user_bundle(reddit_username)
    )
    posts_df = pd.DataFrame(bundle.get("posts") or [])
    logger.info(
        "signup_inference.personality.input.loaded user_id=%s job_id=%s posts=%s",
        user_id,
        job_id,
        len(posts_df),
    )
    if posts_df.empty:
        raise ValueError("Reddit user not found or has no usable data")

    available_traits = _get_available_traits()
    computed_at = datetime.now(timezone.utc)
    if not available_traits:
        raise RuntimeError("No personality model artifacts available for signup inference")

    logger.info(
        "signup_inference.personality.features.prepare_base.start user_id=%s job_id=%s traits=%s",
        user_id,
        job_id,
        available_traits,
    )
    base_posts_df = _build_base_posts_frame(posts_df=posts_df, author=reddit_username)
    logger.info(
        "signup_inference.personality.features.prepare_base.done user_id=%s job_id=%s rows=%s",
        user_id,
        job_id,
        len(base_posts_df),
    )
    predictions_by_trait: dict[str, list[dict[str, Any]]] = {}
    model_versions: dict[str, str] = {}
    trait_errors: dict[str, str] = {}

    for trait in available_traits:
        try:
            logger.info(
                "signup_inference.personality.trait.start user_id=%s job_id=%s trait=%s",
                user_id,
                job_id,
                trait,
            )
            feature_builder = PersonalityInferenceFeatureBuilder(trait=trait, domain="personality")
            prepared_df, feature_columns = feature_builder.prepare(base_posts_df, author=reddit_username)
            logger.info(
                "signup_inference.personality.trait.features.done user_id=%s job_id=%s trait=%s rows=%s features=%s",
                user_id,
                job_id,
                trait,
                len(prepared_df),
                len(feature_columns),
            )
            engine = PersonalityInferenceEngine(
                trait=trait,
                feature_columns=feature_columns,
            )
            prediction_frame = engine.predict(prepared_df)
            confidence_scores = _extract_scores(engine=engine, frame=prepared_df, feature_columns=feature_columns)

            predicted_column = _get_predicted_column(prediction_frame)
            trait_predictions: list[dict[str, Any]] = []
            for index, (_, row) in enumerate(prediction_frame.iterrows()):
                week_value = row.get("week", prepared_df.iloc[index].get("week"))
                trait_predictions.append(
                    {
                        "week_start": _coerce_week_start(week_value),
                        "direction": str(row[predicted_column]),
                        "score": confidence_scores[index],
                    }
                )
            predictions_by_trait[trait] = trait_predictions
            model_versions[trait] = str(engine.model_uri)
            logger.info(
                "signup_inference.personality.trait.done user_id=%s job_id=%s trait=%s predictions=%s",
                user_id,
                job_id,
                trait,
                len(trait_predictions),
            )
        except Exception as exc:
            trait_errors[trait] = str(exc)
            logger.warning("Skipping personality trait '%s' during signup inference: %s", trait, exc)

    if not predictions_by_trait:
        error_summary = "; ".join(f"{trait}: {error}" for trait, error in trait_errors.items()) or "unknown error"
        raise RuntimeError(f"Personality inference failed for all traits: {error_summary}")

    logger.info("signup_inference.personality.db.create_run.start user_id=%s job_id=%s", user_id, job_id)
    run = await create_personality_inference_run(
        db,
        job_id=job_id,
        user_id=user_id,
        reddit_username=reddit_username,
        run_type="signup",
        computed_at=computed_at,
        model_versions=model_versions,
    )
    logger.info(
        "signup_inference.personality.db.create_run.done user_id=%s job_id=%s run_id=%s",
        user_id,
        job_id,
        run.id,
    )

    week_index = _union_week_starts(predictions_by_trait)
    if not week_index:
        error_summary = "; ".join(f"{trait}: {error}" for trait, error in trait_errors.items()) or "unknown error"
        logger.warning(
            "No weekly personality predictions were produced for available traits=%s",
            available_traits,
        )
        raise RuntimeError(f"Personality inference produced no weekly predictions: {error_summary}")

    flattened_predictions: list[dict[str, Any]] = []
    for trait, trait_predictions in predictions_by_trait.items():
        predictions_by_week = {item["week_start"]: item for item in trait_predictions}
        for week_start in week_index:
            prediction = predictions_by_week.get(week_start)
            if prediction is None:
                continue
            flattened_predictions.append(
                {
                    "week_start": week_start,
                    "trait": trait,
                    "direction": prediction["direction"],
                    "score": prediction["score"],
                }
            )

    if not flattened_predictions:
        error_summary = "; ".join(f"{trait}: {error}" for trait, error in trait_errors.items()) or "unknown error"
        raise RuntimeError(f"Personality inference produced no persisted predictions: {error_summary}")

    logger.info(
        "signup_inference.personality.db.insert_predictions.start user_id=%s job_id=%s count=%s",
        user_id,
        job_id,
        len(flattened_predictions),
    )
    await add_personality_weekly_predictions(
        db,
        run_id=run.id,
        user_id=user_id,
        predictions=flattened_predictions,
    )
    logger.info(
        "signup_inference.personality.done user_id=%s job_id=%s predictions=%s",
        user_id,
        job_id,
        len(flattened_predictions),
    )


def _build_base_posts_frame(*, posts_df, author: str):
    """Build base posts frame."""
    from src.personality.inference_feature_builder import PersonalityInferenceFeatureBuilder

    bootstrap_builder = PersonalityInferenceFeatureBuilder(
        trait=PERSONALITY_TRAITS[0],
        domain="personality",
    )
    normalized_df = bootstrap_builder._normalize_input(posts_df.copy(), author=author)
    normalized_df = bootstrap_builder._ensure_processed_text(normalized_df)
    normalized_df = bootstrap_builder._ensure_scores(normalized_df)
    normalized_df = bootstrap_builder._ensure_trait_columns(normalized_df)
    normalized_df = bootstrap_builder._ensure_embeddings(normalized_df)
    return bootstrap_builder._slim_columns(normalized_df)


def _get_available_traits() -> list[str]:
    """Get available traits."""
    return list(PERSONALITY_TRAITS)


def _extract_scores(
    *,
    engine,
    frame,
    feature_columns: list[str],
) -> list[float | None]:
    """Extract scores."""
    model = getattr(engine, "model", None)
    if model is None or not hasattr(model, "predict_proba"):
        return [None] * len(frame)

    probabilities = model.predict_proba(frame[feature_columns].to_numpy())
    try:
        import numpy as np

        probabilities_arr = np.asarray(probabilities)
        if probabilities_arr.ndim == 1:
            return [round(float(value), 4) for value in probabilities_arr.tolist()]
        return [
            round(float(value), 4) if not math.isnan(float(value)) else None
            for value in probabilities_arr.max(axis=1).tolist()
        ]
    except Exception:
        return [None] * len(frame)


def _get_predicted_column(prediction_frame) -> str:
    """Get predicted column."""
    predicted_columns = [column for column in prediction_frame.columns if column.endswith("_predicted")]
    if not predicted_columns:
        raise ValueError("Prediction output did not include a predicted column")
    return predicted_columns[0]


def _coerce_week_start(value: Any):
    """Coerce week start."""
    import pandas as pd

    if isinstance(value, pd.Period):
        return value.start_time.date()

    if isinstance(value, str):
        matches = DATE_PATTERN.findall(value)
        if matches:
            return datetime.fromisoformat(matches[0]).date()
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Unable to parse week value '{value}'")
        return parsed.date()

    if hasattr(value, "start_time"):
        return value.start_time.date()
    if hasattr(value, "date"):
        return value.date()

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse week value '{value}'")
    return parsed.date()


def _union_week_starts(predictions_by_trait: dict[str, list[dict[str, Any]]]) -> list:
    """Handle union week starts."""
    week_sets = [
        {prediction["week_start"] for prediction in predictions}
        for predictions in predictions_by_trait.values()
        if predictions
    ]
    if not week_sets:
        return []
    return sorted(set().union(*week_sets))
