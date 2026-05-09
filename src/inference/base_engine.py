"""Inference utilities for base engine."""
import logging
from typing import List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.inference.artifact_resolver import ArtifactResolver
from src.inference.schema_registry import InferenceSchemaRegistry
from utils.config import get_model_config
from utils.mlflow_registry import configure_mlflow_tracking, resolve_model_uri

logger = logging.getLogger(__name__)


class BaseInferenceEngine:
    """Reusable prediction engine shared across domain-specific inference flows."""

    def __init__(
        self,
        domain: str,
        trait: Optional[str] = None,
        model_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        artifact_resolver: Optional[ArtifactResolver] = None,
        schema_registry: Optional[InferenceSchemaRegistry] = None,
    ):
        """Initialize the base inference engine."""
        self.domain = domain
        self.trait = trait
        self.artifact_resolver = artifact_resolver
        self.schema_registry = schema_registry or InferenceSchemaRegistry()

        schema = self.schema_registry.resolve(
            domain=domain,
            trait=trait,
            feature_columns=feature_columns,
            target_features=target_features,
        )
        self.feature_columns = schema.feature_columns
        self.target_features = schema.target_features
        self.model_uri = self._resolve_model_uri(model_path)
        self.model = None
        self.label_encoder = None

        self._load_model()
        self._load_label_encoder()

    def _resolve_model_uri(self, model_path: Optional[str]) -> str:
        """Resolve model URI."""
        supplied_uri = None
        if model_path:
            supplied_uri = str(model_path).strip()
            if not supplied_uri.startswith("models:/"):
                raise ValueError(
                    "Local model paths are disabled for inference. "
                    "Use an MLflow registry URI (models:/...) or default registry configuration."
                )

        model_key = self._resolve_model_key()
        return resolve_model_uri(model_key=model_key, model_uri=supplied_uri)

    def _resolve_model_key(self) -> str:
        """Resolve model key."""
        if self.domain == "personality":
            if not self.trait:
                raise ValueError("trait is required for personality inference.")
            return f"big5_{self.trait}_state"
        raise ValueError(f"Unsupported domain for BaseInferenceEngine registry mapping: {self.domain}")

    def _load_model(self) -> None:
        """Load model."""
        try:
            configure_mlflow_tracking()
            logger.info("Loading model from MLflow registry URI: %s", self.model_uri)
            self.model = mlflow.sklearn.load_model(self.model_uri)
            logger.info("Model loaded successfully")
            logger.info("Feature columns: %s", self.feature_columns)
            logger.info("Target features: %s", self.target_features)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    def _load_label_encoder(self) -> None:
        """Load label encoder."""
        try:
            label_encoder_path = get_model_config(
                domain=self.domain,
                trait=self.trait,
            ).get("label_encoder_path")
            if not label_encoder_path:
                return

            filename = str(label_encoder_path).replace("\\", "/").split("/")[-1]
            candidates = [filename]
            if filename != "label_encoder.joblib":
                candidates.append("label_encoder.joblib")

            configure_mlflow_tracking()
            for artifact_name in candidates:
                try:
                    artifact_uri = f"{self.model_uri}/{artifact_name}"
                    downloaded = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
                    self.label_encoder = joblib.load(downloaded)
                    logger.info("Label encoder loaded from MLflow artifact %s", artifact_uri)
                    return
                except Exception:
                    continue

            logger.info("No label encoder artifact found for model URI %s", self.model_uri)
        except Exception as exc:
            logger.warning("Failed to load label encoder: %s", exc)
            self.label_encoder = None

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate input."""
        if not self.feature_columns:
            scope = f"{self.domain}:{self.trait}" if self.trait else self.domain
            raise ValueError(f"No feature columns configured for inference scope '{scope}'.")

        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data missing required columns: {missing_cols}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the requested data."""
        try:
            self.validate_input(df)
            features = df[self.feature_columns].to_numpy()
            logger.info("Making predictions for %s samples", len(features))

            predictions = self.model.predict(features)
            predictions_arr = np.asarray(predictions)
            if predictions_arr.ndim == 1:
                predictions_arr = predictions_arr.reshape(-1, 1)
            predictions_arr = self._decode_predictions(predictions_arr)

            result_df = df.copy()
            default_target = self.target_features[0] if self.target_features else "prediction"
            for index in range(predictions_arr.shape[1]):
                target_name = (
                    self.target_features[index]
                    if index < len(self.target_features)
                    else f"{default_target}_{index}"
                )
                result_df[f"{target_name}_predicted"] = predictions_arr[:, index]

            logger.info("Predictions completed for %s samples", len(result_df))
            return result_df
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise

    def format_output(
        self,
        predictions_df: pd.DataFrame,
        include_features: bool = False,
    ) -> pd.DataFrame:
        """Format output."""
        predicted_cols = [col for col in predictions_df.columns if col.endswith("_predicted")]

        display_cols = []
        for col in ["author", "week"]:
            if col in predictions_df.columns:
                display_cols.append(col)

        if include_features:
            for col in self.feature_columns:
                if col in predictions_df.columns and col not in display_cols:
                    display_cols.append(col)

        display_cols.extend(predicted_cols)
        return predictions_df[display_cols]

    def _decode_predictions(self, predictions_arr: np.ndarray) -> np.ndarray:
        """Decode predictions."""
        if self.label_encoder is None or predictions_arr.shape[1] != 1:
            return predictions_arr

        raw_values = predictions_arr[:, 0]
        try:
            if not np.issubdtype(raw_values.dtype, np.number):
                return predictions_arr
        except TypeError:
            return predictions_arr

        rounded = np.rint(raw_values).astype(int)
        if not np.allclose(raw_values.astype(float), rounded.astype(float)):
            return predictions_arr

        classes = getattr(self.label_encoder, "classes_", None)
        if classes is None or len(classes) == 0:
            return predictions_arr
        if rounded.min() < 0 or rounded.max() >= len(classes):
            return predictions_arr

        decoded = self.label_encoder.inverse_transform(rounded)
        return np.asarray(decoded, dtype=object).reshape(-1, 1)
