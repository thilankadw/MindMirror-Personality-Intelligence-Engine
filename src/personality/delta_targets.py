"""Utilities for delta targets."""
import logging
from typing import List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StdNaNHandler:
    """Handle NaN values in standard deviation columns."""
    
    def __init__(self, trait_columns: List[str], std_cols: List[str]):
        """Initialize the std na n handler."""
        self.trait_columns = trait_columns
        self.std_cols = std_cols
    
    def handle(self, df: pd.DataFrame, groupby_col: str = 'author') -> pd.DataFrame:
        """
        Handle NaN values in std columns.
        
        First fills with user-level mean, then with global mean for users
        who only posted once per week (making std calculation impossible).
        
        Args:
            df: DataFrame with std columns
            groupby_col: Column to group by (default: 'author')
            
        Returns:
            DataFrame with NaN values filled
        """
        try:
            logger.info("Handling NaN values in std columns...")
            df_copy = df.copy()
            
            for col in self.std_cols:
                # First pass: fill with user-level mean
                df_copy[col] = df_copy.groupby(groupby_col)[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                
                # Second pass: fill remaining NaN with global mean
                global_mean = df_copy[col].mean()
                nan_count = df_copy[col].isna().sum()
                
                if nan_count > 0:
                    logger.info(f"Filling {nan_count} remaining NaN in {col} with global mean: {global_mean:.6f}")
                    df_copy[col] = df_copy[col].fillna(global_mean)
            
            logger.info("NaN handling completed")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error handling NaN values: {str(e)}")
            raise


class TraitDeltaTargetCreator:
    """Create delta target columns for personality trait predictions."""
    
    def __init__(
        self,
        trait_columns: List[str],
        mean_cols: List[str],
        time_column: str,
    ):
        """Initialize the trait delta target creator."""
        self.trait_columns = trait_columns
        self.mean_cols = mean_cols
        self.time_column = time_column
    
    def create(self, df: pd.DataFrame, groupby_col: str = 'author') -> pd.DataFrame:
        """
        Create delta target columns by computing weekly changes in trait scores.
        
        Delta = Current week score - Previous week score (for same author)
        
        Args:
            df: DataFrame with weekly aggregated trait scores
            groupby_col: Column to group by (default: 'author')
            
        Returns:
            DataFrame with delta columns added
        """
        try:
            logger.info("Creating delta target columns...")
            df_copy = df.copy()
            
            # Sort by author and time to ensure proper temporal ordering
            if self.time_column not in df_copy.columns:
                raise ValueError(f"Missing time column: {self.time_column}")
            df_copy = df_copy.sort_values([groupby_col, self.time_column]).reset_index(drop=True)
            logger.info("Data sorted by %s and %s", groupby_col, self.time_column)
            
            # Create delta columns for each trait
            delta_cols_created = []
            for trait_col in self.mean_cols:
                trait_name = trait_col.replace('_mean_w', '')
                delta_col = f'delta_{trait_name}'
                
                # Calculate difference from previous week for each author
                df_copy[delta_col] = df_copy.groupby(groupby_col)[trait_col].diff()
                delta_cols_created.append(delta_col)
                
                # Log statistics
                non_null_count = df_copy[delta_col].notna().sum()
                null_count = df_copy[delta_col].isna().sum()
                logger.info(f"Created {delta_col}: {non_null_count} observations, {null_count} NaN (first week)")
            
            logger.info(f"Delta target creation completed. New columns: {delta_cols_created}")
            logger.info(f"Final shape: {df_copy.shape}")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error creating delta targets: {str(e)}")
            raise


class FutureDeltaTargetCreator:
    """Create future-vs-past delta targets using rolling windows."""

    def __init__(
        self,
        obs_column: str,
        past_window: int,
        future_window: int,
        time_column: str,
        past_mean_col: str = "past_mean",
        future_mean_col: str = "future_mean",
        delta_col: str = "delta_future",
    ):
        """Initialize the future delta target creator."""
        self.obs_column = obs_column
        self.past_window = past_window
        self.future_window = future_window
        self.time_column = time_column
        self.past_mean_col = past_mean_col
        self.future_mean_col = future_mean_col
        self.delta_col = delta_col

    def create(self, df: pd.DataFrame, groupby_col: str = "author") -> pd.DataFrame:
        """Create the requested data."""
        try:
            logger.info("Creating future delta target columns...")
            df_copy = df.copy()

            if self.time_column not in df_copy.columns:
                raise ValueError(f"Missing time column: {self.time_column}")
            df_copy = df_copy.sort_values([groupby_col, self.time_column]).reset_index(drop=True)

            df_copy[self.past_mean_col] = df_copy.groupby(groupby_col)[self.obs_column].transform(
                lambda s: s.shift(1).rolling(self.past_window, min_periods=self.past_window).mean()
            )
            df_copy[self.future_mean_col] = df_copy.groupby(groupby_col)[self.obs_column].transform(
                lambda s: s.rolling(self.future_window, min_periods=self.future_window).mean().shift(-self.future_window)
            )
            df_copy[self.delta_col] = df_copy[self.future_mean_col] - df_copy[self.past_mean_col]

            nulls = {
                self.past_mean_col: int(df_copy[self.past_mean_col].isna().sum()),
                self.future_mean_col: int(df_copy[self.future_mean_col].isna().sum()),
                self.delta_col: int(df_copy[self.delta_col].isna().sum()),
            }
            logger.info(
                "Future delta targets created. Columns: %s, %s, %s",
                self.past_mean_col,
                self.future_mean_col,
                self.delta_col,
            )
            logger.info("Future delta nulls: %s", nulls)
            return df_copy

        except Exception as e:
            logger.error("Error creating future delta targets: %s", str(e))
            raise


class DirectionLabeler:
    """Assign direction labels based on delta magnitude and a quantile threshold."""

    def __init__(
        self,
        delta_column: str = "delta_future",
        label_column: str = "direction_label",
        tau_quantile: float = 0.7,
        train_mask_column: str = "is_train",
    ):
        """Initialize the direction labeler."""
        self.delta_column = delta_column
        self.label_column = label_column
        self.tau_quantile = tau_quantile
        self.train_mask_column = train_mask_column
        self.tau_ = None

    def fit(self, df: pd.DataFrame) -> float:
        """Fit the requested data."""
        if self.train_mask_column not in df.columns:
            raise ValueError(f"Missing train mask column: {self.train_mask_column}")
        train_df = df[df[self.train_mask_column]].copy()
        if train_df.empty:
            raise ValueError("Training subset is empty. Cannot compute tau.")
        self.tau_ = train_df[self.delta_column].abs().quantile(self.tau_quantile)
        logger.info("Computed tau=%.6f at quantile %.2f", self.tau_, self.tau_quantile)
        return self.tau_

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the requested data."""
        if self.tau_ is None:
            raise ValueError("tau_ is not set. Call fit() first.")
        df_copy = df.copy()
        df_copy[self.label_column] = df_copy[self.delta_column].apply(
            lambda x: self._direction_label(x, self.tau_)
        )
        label_counts = df_copy[self.label_column].value_counts(dropna=False).to_dict()
        logger.info("Direction labels assigned: %s", label_counts)
        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the input data."""
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def _direction_label(value: float, threshold: float) -> str:
        """Handle direction label."""
        if value > threshold:
            return "up"
        if value < -threshold:
            return "down"
        return "neutral"
