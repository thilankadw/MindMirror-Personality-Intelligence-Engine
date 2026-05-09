"""Utilities for smote preprocessing."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImbalancePreprocessor(ABC):
    """Preprocess imbalance data."""
    @abstractmethod
    def fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """Resample training features and labels."""
        pass


class SMOTEPreprocessor(ImbalancePreprocessor):
    """Preprocess smote data."""
    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        random_state: int = 42,
        k_neighbors: int = 5,
    ):
        """Initialize the smote preprocessor."""
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

        try:
            from imblearn.over_sampling import SMOTE
        except Exception as e:
            raise ImportError(
                "SMOTE requires imbalanced-learn. Install with `pip install imbalanced-learn`."
            ) from e

        self._smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )

    def fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """Apply SMOTE on training data only and return resampled X, y."""
        try:
            logger.info(
                "Applying SMOTE: sampling_strategy=%s random_state=%s k_neighbors=%s",
                self.sampling_strategy,
                self.random_state,
                self.k_neighbors,
            )

            x_is_df = isinstance(X, pd.DataFrame)
            y_is_series = isinstance(y, pd.Series)
            x_columns = X.columns.tolist() if x_is_df else None
            y_name = y.name if y_is_series else None

            X_np = X.to_numpy() if x_is_df else np.asarray(X)
            y_np = y.to_numpy() if y_is_series else np.asarray(y)

            if y_np.ndim > 1:
                y_np = y_np.ravel()

            X_res, y_res = self._smote.fit_resample(X_np, y_np)

            if x_is_df:
                X_res = pd.DataFrame(X_res, columns=x_columns)
            if y_is_series:
                y_res = pd.Series(y_res, name=y_name)

            logger.info(
                "SMOTE completed: before=(%s, %s) after=(%s, %s)",
                X_np.shape,
                y_np.shape,
                np.asarray(X_res).shape,
                np.asarray(y_res).shape,
            )
            return X_res, y_res
        except Exception as e:
            logger.error("Error applying SMOTE: %s", str(e))
            raise

    def resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """Alias for fit_resample."""
        return self.fit_resample(X, y)
