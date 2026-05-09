"""Utilities for data splitter."""
import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSplittingStrategy(ABC):
    """Implement data splitting behavior."""
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) ->Tuple[pd.
        DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data."""
        pass

class SplitType(str, Enum):
    """Represent split type."""
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'

class SimpleTrainTextSplittingStrategy(DataSplittingStrategy):
    """Implement simple train text splitting behavior."""
    def __init__(self, test_size = 0.2):
        """Initialize the simple train text splitting strategy."""
        self.test_size = test_size
    
    def split_data(self, df, feature_columns, target_columns):
        """Split data."""
        Y = df[target_columns].values
        X = df[feature_columns].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size)

        return X_train, X_test, Y_train, Y_test


class AuthorTimeSplitStrategy(DataSplittingStrategy):
    """Author-wise forward split that preserves temporal order within each author."""

    def __init__(
        self,
        test_size: float = 0.2,
        author_col: str = "author",
        time_col: str = None,
        drop_small_groups: bool = True,
        min_rows: int = 2,
        use_ceil_for_test: bool = True,
    ):
        """Initialize the author time split strategy."""
        self.test_size = test_size
        self.author_col = author_col
        self.time_col = time_col
        self.drop_small_groups = drop_small_groups
        self.min_rows = min_rows
        self.use_ceil_for_test = use_ceil_for_test

    def split_data(self, df, feature_columns, target_columns):
        """Split data."""
        df_copy = df.copy()
        time_col = self._resolve_time_column(df_copy.columns)
        if time_col:
            df_copy = df_copy.sort_values([self.author_col, time_col])
        else:
            df_copy = df_copy.sort_values([self.author_col])

        train_idx = []
        test_idx = []
        skipped = []

        for author, g in df_copy.groupby(self.author_col):
            g = g.sort_values(time_col) if time_col else g
            n = len(g)
            if self.drop_small_groups and n < self.min_rows:
                skipped.append(author)
                continue
            if n == 0:
                continue

            if self.use_ceil_for_test:
                n_test = int(np.ceil(n * self.test_size))
                if n_test >= n:
                    n_test = 1
                n_train = n - n_test
            else:
                n_train = int(np.floor(n * (1 - self.test_size)))
                if n > 1:
                    n_train = min(max(n_train, 1), n - 1)
                else:
                    n_train = n
                n_test = n - n_train

            train_idx.extend(g.index[:n_train])
            test_idx.extend(g.index[n_train:])

        if skipped:
            logging.info("Skipped authors with < %s rows: %s", self.min_rows, len(skipped))

        X_train = df_copy.loc[train_idx, feature_columns].values
        X_test = df_copy.loc[test_idx, feature_columns].values
        Y_train = df_copy.loc[train_idx, target_columns].values
        Y_test = df_copy.loc[test_idx, target_columns].values

        return X_train, X_test, Y_train, Y_test

    def split_mask(self, df) -> pd.Series:
        """Split mask."""
        df_copy = df.copy()
        time_col = self._resolve_time_column(df_copy.columns)
        if time_col:
            df_copy = df_copy.sort_values([self.author_col, time_col])
        else:
            df_copy = df_copy.sort_values([self.author_col])

        train_idx = []
        test_idx = []
        skipped = []

        for author, g in df_copy.groupby(self.author_col):
            g = g.sort_values(time_col) if time_col else g
            n = len(g)
            if self.drop_small_groups and n < self.min_rows:
                skipped.append(author)
                continue
            if n == 0:
                continue

            if self.use_ceil_for_test:
                n_test = int(np.ceil(n * self.test_size))
                if n_test >= n:
                    n_test = 1
                n_train = n - n_test
            else:
                n_train = int(np.floor(n * (1 - self.test_size)))
                if n > 1:
                    n_train = min(max(n_train, 1), n - 1)
                else:
                    n_train = n
                n_test = n - n_train

            train_idx.extend(g.index[:n_train])
            test_idx.extend(g.index[n_train:])

        if skipped:
            logging.info("Skipped authors with < %s rows: %s", self.min_rows, len(skipped))

        mask = pd.Series(False, index=df_copy.index)
        mask.loc[train_idx] = True
        logging.info(
            "AuthorTimeSplit: train=%s test=%s total=%s",
            int(mask.sum()),
            int((~mask).sum()),
            len(mask),
        )
        return mask

    def add_split_column(self, df, column: str = "is_train") -> pd.DataFrame:
        """Add split column."""
        df_copy = df.copy()
        df_copy[column] = self.split_mask(df_copy)
        return df_copy

    def _resolve_time_column(self, columns) -> str:
        """Resolve time column."""
        if not self.time_col:
            raise ValueError("time_col must be provided for AuthorTimeSplitStrategy.")
        if self.time_col not in columns:
            raise ValueError(f"Missing time column: {self.time_col}")
        return self.time_col
