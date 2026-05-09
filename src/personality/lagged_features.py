"""Utilities for lagged features."""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LaggedFeatureCreator:
    """
    Creates lagged features for time series data to support multi-timestep models.
    For timesteps=2, each sample will contain features from [t-1, t].
    For timesteps=3, each sample will contain features from [t-2, t-1, t].
    """
    
    def __init__(
        self,
        timesteps: int = 1,
        feature_columns: list = None,
        group_column: str = "author",
        time_column: str = None,
    ):
        """
        Initialize the lagged feature creator.
        
        Args:
            timesteps: Number of time steps to include (1 = no lags, 2 = 1 lag, etc.)
            feature_columns: List of feature column names to create lags for
            group_column: Column to group by (e.g., 'author')
        """
        self.timesteps = timesteps
        self.feature_columns = feature_columns or []
        self.group_column = group_column
        self.time_column = time_column
        
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features based on timesteps configuration.
        
        Args:
            df: Input dataframe with time series data
            
        Returns:
            DataFrame with lagged features (flattened format)
        """
        if self.timesteps == 1:
            logger.info("timesteps=1, no lagged features needed")
            return df
        
        if not self.feature_columns:
            raise ValueError("feature_columns must be specified for lagged feature creation")
        
        logger.info(f"Creating lagged features with timesteps={self.timesteps}")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        # Sort by group and time if available
        time_col = self._resolve_time_column(df.columns)
        if time_col:
            df = df.sort_values([self.group_column, time_col]).reset_index(drop=True)
        else:
            df = df.sort_values([self.group_column]).reset_index(drop=True)
        
        # Create lagged features
        lagged_dfs = []
        
        # For each lag from (timesteps-1) to 0 (creating features in chronological order)
        # For timesteps=2: create lag 1 (t-1) then lag 0 (t), resulting in [t-1, t]
        for lag in range(self.timesteps - 1, -1, -1):
            if lag == 0:
                # Current timestep (t) - no shift needed
                lag_df = df[self.feature_columns].copy()
                lag_df.columns = [f'{col}_t{self.timesteps - 1}' for col in self.feature_columns]
            else:
                # Lagged timesteps (t-lag)
                lag_df = df.groupby(self.group_column)[self.feature_columns].shift(lag)
                lag_df.columns = [f'{col}_t{self.timesteps - 1 - lag}' for col in self.feature_columns]
            
            lagged_dfs.append(lag_df)
        
        # Concatenate all lagged features in chronological order
        lagged_features = pd.concat(lagged_dfs, axis=1)
        
        # Identify columns to preserve (non-feature columns that shouldn't be lagged)
        preserve_cols = [
            col
            for col in df.columns
            if col not in self.feature_columns
            and not col.startswith("delta_")
            and col not in [self.group_column, time_col]
        ]
        
        # Add back metadata columns
        base_cols = [self.group_column]
        if time_col and time_col in df.columns:
            base_cols.append(time_col)

        result_df = pd.concat(
            [
                df[base_cols],
                lagged_features,
                df[preserve_cols] if preserve_cols else pd.DataFrame(),
            ],
            axis=1,
        )
        
        # Add target columns if they exist
        target_cols = [col for col in df.columns if col.startswith('delta_')]
        if target_cols:
            result_df = pd.concat([result_df, df[target_cols]], axis=1)
        
        # Drop rows with NaN (first timesteps-1 rows for each group)
        rows_before = len(result_df)
        result_df = result_df.dropna()
        rows_after = len(result_df)
        
        logger.info(f"Lagged features created. Dropped {rows_before - rows_after} rows due to insufficient history")
        logger.info(f"Result shape: {result_df.shape}")
        
        # Log sample to verify ordering
        if len(result_df) > 0:
            sample_cols = [col for col in result_df.columns if 'Agreeableness_mean_w' in col]
            if sample_cols:
                logger.info(f"Sample feature ordering (Agreeableness_mean_w): {sample_cols}")
        
        return result_df
    
    def get_lagged_feature_names(self) -> list:
        """
        Get the names of all lagged feature columns that will be created.
        
        Returns:
            List of lagged feature column names in chronological order [t-n+1, ..., t-1, t]
        """
        feature_names = []
        # Create feature names in chronological order
        for t in range(self.timesteps):
            for col in self.feature_columns:
                feature_names.append(f'{col}_t{t}')
        return feature_names

    def _resolve_time_column(self, columns) -> str:
        """Resolve time column."""
        if not self.time_column:
            return None
        if self.time_column not in columns:
            raise ValueError(f"Missing time column: {self.time_column}")
        return self.time_column


class RollingEmbeddingFeatureCreator:
    """Compute rolling mean embeddings from previous K posts (no current embedding)."""

    def __init__(
        self,
        embedding_column: str = "text_embeddings_mean_w",
        group_column: str = "author",
        time_column: str = None,
        window: int = 3,
        output_column: str = "roll_emb",
    ):
        """Initialize the rolling embedding feature creator."""
        self.embedding_column = embedding_column
        self.group_column = group_column
        self.time_column = time_column
        self.window = window
        self.output_column = output_column

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the requested data."""
        df_copy = df.copy()
        time_col = self._resolve_time_column(df_copy.columns)
        if time_col:
            df_copy = df_copy.sort_values([self.group_column, time_col]).reset_index(drop=True)
        else:
            df_copy = df_copy.sort_values([self.group_column]).reset_index(drop=True)

        df_copy[self.output_column] = df_copy.groupby(
            self.group_column, group_keys=False
        ).apply(lambda g: self._compute_prev_emb(g[self.embedding_column]))

        return df_copy

    def expand_embeddings(self, df: pd.DataFrame, source_column: str = None, prefix: str = "prev_emb_"):
        """Expand embeddings."""
        col = source_column or self.output_column
        if col not in df.columns:
            raise ValueError(f"Embedding column not found: {col}")

        first_valid = df[col].dropna().iloc[0]
        emb_dim = len(first_valid)
        emb_cols = [f"{prefix}{i}" for i in range(emb_dim)]
        emb_matrix = np.vstack(df[col].tolist())
        emb_df = pd.DataFrame(emb_matrix, index=df.index, columns=emb_cols)
        return pd.concat([df, emb_df], axis=1), emb_cols

    def _compute_prev_emb(self, series: pd.Series):
        """Handle compute prev emb."""
        embs = series.tolist()
        roll = [None] * len(embs)
        for i in range(len(embs)):
            if i < self.window:
                roll[i] = None
                continue
            window = embs[i - self.window : i]
            if any(not isinstance(x, np.ndarray) for x in window):
                roll[i] = None
                continue
            roll[i] = np.mean(np.vstack(window), axis=0)
        return pd.Series(roll, index=series.index)

    def _resolve_time_column(self, columns) -> str:
        """Resolve time column."""
        if not self.time_column:
            return None
        if self.time_column not in columns:
            raise ValueError(f"Missing time column: {self.time_column}")
        return self.time_column


class PreviousStateFeatureCreator:
    """Create lagged target features like mean over past K and t-1."""

    def __init__(
        self,
        target_column: str,
        group_column: str = "author",
        window: int = 3,
        mean_output: str = "prev_state_meanK",
        lag_output: str = "prev_state_1",
    ):
        """Initialize the previous state feature creator."""
        self.target_column = target_column
        self.group_column = group_column
        self.window = window
        self.mean_output = mean_output
        self.lag_output = lag_output

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the requested data."""
        df_copy = df.copy()
        df_copy[self.mean_output] = df_copy.groupby(self.group_column)[
            self.target_column
        ].transform(lambda s: s.shift(1).rolling(self.window, min_periods=self.window).mean())
        df_copy[self.lag_output] = df_copy.groupby(self.group_column)[
            self.target_column
        ].shift(1)
        return df_copy


class FeatureShiftCreator:
    """Shift feature columns within each group to enforce causality."""

    def __init__(
        self,
        feature_columns: list,
        group_column: str = "author",
        lag: int = 1,
        output_suffix: str = None,
    ):
        """Initialize the feature shift creator."""
        self.feature_columns = feature_columns
        self.group_column = group_column
        self.lag = lag
        self.output_suffix = output_suffix

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the requested data."""
        df_copy = df.copy()
        if self.output_suffix:
            for col in self.feature_columns:
                df_copy[f"{col}{self.output_suffix}"] = df_copy.groupby(self.group_column)[col].shift(self.lag)
        else:
            df_copy[self.feature_columns] = df_copy.groupby(self.group_column)[self.feature_columns].shift(self.lag)
        nulls = {col: int(df_copy[col].isna().sum()) for col in self.feature_columns}
        logger.info("Shifted features by %s: %s", self.lag, nulls)
        return df_copy


class RollingStatsFeatureCreator:
    """Create rolling mean/std features within each group."""

    def __init__(
        self,
        value_column: str,
        group_column: str = "author",
        window: int = 4,
        min_periods: int = 2,
        mean_column: str = "state_roll4",
        std_column: str = "state_roll4_std",
    ):
        """Initialize the rolling stats feature creator."""
        self.value_column = value_column
        self.group_column = group_column
        self.window = window
        self.min_periods = min_periods
        self.mean_column = mean_column
        self.std_column = std_column

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the requested data."""
        df_copy = df.copy()
        df_copy[self.mean_column] = df_copy.groupby(self.group_column)[self.value_column].transform(
            lambda s: s.rolling(self.window, min_periods=self.min_periods).mean()
        )
        df_copy[self.std_column] = df_copy.groupby(self.group_column)[self.value_column].transform(
            lambda s: s.rolling(self.window, min_periods=self.min_periods).std()
        )
        mean_nulls = int(df_copy[self.mean_column].isna().sum())
        std_nulls = int(df_copy[self.std_column].isna().sum())
        logger.info(
            "Rolling stats created: %s nulls=%s, %s nulls=%s",
            self.mean_column,
            mean_nulls,
            self.std_column,
            std_nulls,
        )
        return df_copy


class LagFeatureCreator:
    """Create a single lagged feature column within each group."""

    def __init__(
        self,
        value_column: str,
        group_column: str = "author",
        lag: int = 1,
        output_column: str = "state_lag1",
    ):
        """Initialize the lag feature creator."""
        self.value_column = value_column
        self.group_column = group_column
        self.lag = lag
        self.output_column = output_column

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the requested data."""
        df_copy = df.copy()
        df_copy[self.output_column] = df_copy.groupby(self.group_column)[self.value_column].shift(self.lag)
        nulls = int(df_copy[self.output_column].isna().sum())
        logger.info("Lag feature created: %s nulls=%s", self.output_column, nulls)
        return df_copy
