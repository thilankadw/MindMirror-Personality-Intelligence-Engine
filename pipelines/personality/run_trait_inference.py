"""Pipeline helpers for run trait inference."""
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.personality.inference_engine import PersonalityInferenceEngine
from src.personality.inference_feature_builder import PersonalityInferenceFeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PostsInput = Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any], str]


def run_trait_inference(
    trait: str,
    posts: PostsInput,
    author: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    include_features: bool = False,
    gap_weeks: float = 0.0,
) -> pd.DataFrame:
    """Run trait inference."""
    try:
        logger.info("=" * 80)
        logger.info("Weekly Trait Change Prediction - Inference Pipeline (%s)", trait)
        logger.info("=" * 80)

        df = load_posts(posts)
        logger.info("Loaded %s posts", len(df))

        feature_builder = PersonalityInferenceFeatureBuilder(
            trait=trait,
            domain="personality",
        )
        prepared_df, feature_columns = feature_builder.prepare(
            posts_df=df,
            author=author,
            gap_weeks=gap_weeks,
        )
        inference_engine = PersonalityInferenceEngine(
            trait=trait,
            model_path=model_path,
            feature_columns=feature_columns,
        )

        logger.info("Making predictions...")
        predictions_df = inference_engine.predict(prepared_df)
        output_df = inference_engine.format_output(
            predictions_df,
            include_features=include_features,
        )

        destination = output_path or "artifacts/predictions/output/predictions.json"
        save_predictions(
            predictions_df=output_df,
            output_path=destination,
            group_field=feature_builder.author_col,
        )
        logger.info("Predictions saved to: %s", destination)
        logger.info("Inference Pipeline Completed Successfully")
        return output_df
    except Exception as exc:
        logger.error("Inference pipeline failed for %s: %s", trait, exc)
        raise


def load_posts(posts: PostsInput) -> pd.DataFrame:
    """Load posts."""
    logger.info("Loading input posts...")

    if isinstance(posts, str):
        if posts.endswith(".csv"):
            return pd.read_csv(posts)
        if posts.endswith(".json"):
            with open(posts, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            return _coerce_posts_payload(payload)
        raise ValueError("Input file must be CSV or JSON")

    if isinstance(posts, dict):
        return _coerce_posts_payload(posts)

    if isinstance(posts, list):
        return _coerce_posts_payload(posts)

    if isinstance(posts, pd.DataFrame):
        df = posts.copy()
        if (
            "posts" in df.columns
            and len(df) == 1
            and isinstance(df["posts"].iloc[0], list)
        ):
            payload = df.iloc[0].to_dict()
            return _coerce_posts_payload(payload)
        return df

    raise ValueError("posts must be DataFrame, list of dicts, or file path")


def _coerce_posts_payload(payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
    """Coerce posts payload."""
    if isinstance(payload, dict):
        if "posts" in payload and isinstance(payload["posts"], list):
            df = pd.DataFrame(payload["posts"])
            author = _extract_author(payload)
            if author and "author" not in df.columns:
                if "provider_username" in df.columns:
                    df["author"] = df["provider_username"].fillna(author)
                else:
                    df["author"] = author
            return df
        return pd.DataFrame([payload])

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    raise ValueError("Unsupported JSON payload format for posts input")


def _extract_author(payload: Dict[str, Any]) -> Optional[str]:
    """Extract author."""
    for field in ["platform_username", "external_username", "provider_username", "username"]:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: str,
    group_field: str = "author",
) -> None:
    """Save predictions."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if output_path.endswith(".json"):
            result = {}
            if group_field in predictions_df.columns and not predictions_df.empty:
                result[group_field] = predictions_df[group_field].iloc[0]

            predictions = []
            predicted_cols = [
                col for col in predictions_df.columns if col.endswith("_predicted")
            ]

            for _, row in predictions_df.iterrows():
                pred_dict = {}
                for time_col in ["week", "week_start", "week_end"]:
                    if time_col in predictions_df.columns:
                        pred_dict[time_col] = str(row[time_col])

                for col in predicted_cols:
                    clean_name = col.replace("_predicted", "")
                    value = row[col]
                    pred_dict[clean_name] = value.item() if hasattr(value, "item") else value

                predictions.append(pred_dict)

            result["predictions"] = predictions
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(result, file_handle, indent=2)
        else:
            predictions_df.to_csv(output_path, index=False)

        logger.info("Saved %s predictions to %s", len(predictions_df), output_path)
    except Exception as exc:
        logger.error("Failed to save predictions: %s", exc)
        raise


def build_inference_parser() -> argparse.ArgumentParser:
    """Build inference parser."""
    parser = argparse.ArgumentParser(
        description="Predict personality trait changes for next week based on current week posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--posts",
        type=str,
        required=True,
        help="Path to posts data (CSV or JSON) for one user for one week",
    )
    parser.add_argument(
        "--author",
        type=str,
        help="Author name (required if not in data)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Optional MLflow registry URI (models:/...) to override the configured model.",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        type=str,
        help="Path to save predictions (CSV or JSON)",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Include input features in output",
    )
    parser.add_argument(
        "--gap-weeks",
        type=float,
        default=0.0,
        help="Time gap since last week (default: 0.0)",
    )
    return parser
