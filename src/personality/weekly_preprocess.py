"""Utilities for weekly preprocess."""
import logging
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
import ast
import re

logger = logging.getLogger(__name__)

class WeeklyAggregator(ABC):
    """Abstract base class for weekly aggregation operations."""
    
    @abstractmethod
    def aggregate(self, df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
        """Aggregate data weekly."""
        pass

class TraitWeeklyAggregator(WeeklyAggregator):
    """Aggregate personality trait scores by week."""
    
    def __init__(
        self,
        trait_columns: List[str],
        created_date_col: str = "created_date",
        week_col: str = "week",
    ):
        """Initialize the trait weekly aggregator."""
        self.trait_columns = trait_columns
        self.created_date_col = created_date_col
        self.week_col = week_col
    
    def aggregate(
        self,
        df: pd.DataFrame,
        groupby_cols: List[str] = None,
    ) -> pd.DataFrame:
        """Aggregate the requested data."""
        try:
            logger.info(f"Aggregating {len(self.trait_columns)} traits weekly...")

            if groupby_cols is None:
                groupby_cols = ["author", self.week_col]

            if self.week_col not in df.columns:
                if self.created_date_col not in df.columns:
                    raise ValueError(
                        f"Missing '{self.created_date_col}' required to create '{self.week_col}'."
                    )
                logger.info("Creating week column...")
                df[self.created_date_col] = pd.to_datetime(df[self.created_date_col])
                df[self.week_col] = df[self.created_date_col].dt.to_period('W')
                logger.info("Week column created...")
            
            weekly_agg_list = []
            
            for trait in self.trait_columns:
                # Calculate mean
                trait_mean = df.groupby(groupby_cols)[trait].mean().reset_index()
                trait_mean.rename(columns={trait: f'{trait}_mean_w'}, inplace=True)
                
                # Calculate std
                trait_std = df.groupby(groupby_cols)[trait].std().reset_index()
                trait_std.rename(columns={trait: f'{trait}_std_w'}, inplace=True)
                
                # Merge mean and std
                trait_agg = trait_mean.merge(trait_std, on=groupby_cols)
                weekly_agg_list.append(trait_agg)
            
            # Merge all traits
            weekly_traits = weekly_agg_list[0]
            for trait_agg in weekly_agg_list[1:]:
                weekly_traits = weekly_traits.merge(trait_agg, on=groupby_cols)
            
            logger.info(f"Traits aggregated. Shape: {weekly_traits.columns.tolist()}")
            return weekly_traits
            
        except Exception as e:
            logger.error(f"Error aggregating traits: {str(e)}")
            raise

class EmbeddingWeeklyAggregator(WeeklyAggregator):
    """Aggregate text embeddings by week."""

    def __init__(
        self,
        embedding_column: str = "text_embeddings",
        output_column: str = "text_embeddings_mean_w",
        created_date_col: str = "created_date",
        week_col: str = "week",
    ):
        """Initialize the embedding weekly aggregator."""
        self.embedding_column = embedding_column
        self.output_column = output_column
        self.created_date_col = created_date_col
        self.week_col = week_col

    def aggregate(
        self,
        df: pd.DataFrame,
        groupby_cols: List[str] = None,
    ) -> pd.DataFrame:
        """Aggregate the requested data."""
        try:
            logger.info("Aggregating text embeddings weekly...")
            
            if groupby_cols is None:
                groupby_cols = ["author", self.week_col]

            # Check if week column exists, if not create it
            if self.week_col not in df.columns and self.created_date_col in df.columns:
                logger.info("Creating week column for embeddings...")
                df[self.created_date_col] = pd.to_datetime(df[self.created_date_col])
                df[self.week_col] = df[self.created_date_col].dt.to_period('W')
            elif self.week_col not in df.columns:
                raise ValueError(
                    f"Missing '{self.week_col}' and '{self.created_date_col}' columns for weekly aggregation."
                )
            
            weekly_embeddings = df.groupby(groupby_cols)[self.embedding_column].apply(
                self._mean_embeddings
            ).reset_index()
            weekly_embeddings.rename(columns={self.embedding_column: self.output_column}, inplace=True)
            
            logger.info(f"Embeddings aggregated. Shape: {weekly_embeddings.shape}")
            return weekly_embeddings
            
        except Exception as e:
            logger.error(f"Error aggregating embeddings: {str(e)}")
            raise
    
    def _mean_embeddings(self, embeddings):
        """Calculate mean of embeddings."""
        return np.mean(np.vstack(embeddings.values), axis=0)


class WeeklyFeatureCalculator(ABC):
    """Abstract base class for weekly feature calculation operations."""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly features."""
        pass


class GapWeekCalculator(WeeklyFeatureCalculator):
    """Calculate time-gap features between consecutive weeks for each author."""

    def __init__(
        self,
        author_col: str = "author",
        week_col: str = "week",
        time_col: str = "week_date",
    ):
        """Initialize the gap week calculator."""
        self.author_col = author_col
        self.week_col = week_col
        self.time_col = time_col

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the requested data."""
        try:
            logger.info("Calculating time-gap features...")
            df_copy = df.copy()

            if self.time_col not in df_copy.columns:
                if self.week_col not in df_copy.columns:
                    raise ValueError(
                        f"Missing '{self.week_col}' and '{self.time_col}' columns for time-gap features."
                    )
                df_copy[self.time_col] = df_copy[self.week_col].apply(self._parse_week_value)
            null_week_dates = int(df_copy[self.time_col].isna().sum())
            if null_week_dates > 0:
                logger.info("%s nulls: %s", self.time_col, null_week_dates)

            df_copy = df_copy.sort_values([self.author_col, self.time_col]).reset_index(drop=True)

            df_copy["time_gap_weeks"] = (
                df_copy.groupby(self.author_col)[self.time_col].diff().dt.days / 7.0
            )
            df_copy["log_time_gap_weeks"] = np.log1p(df_copy["time_gap_weeks"])

            # Backwards-compatible column name used in config
            df_copy["gap_weeks"] = df_copy["time_gap_weeks"]

            stats = df_copy["time_gap_weeks"].describe()
            logger.info("Time-gap features calculated. Shape: %s", df_copy.shape)
            logger.info("time_gap_weeks stats: %s", stats.to_dict())
            return df_copy

        except Exception as e:
            logger.error(f"Error calculating gap weeks: {str(e)}")
            raise

    def _parse_week_value(self, val):
        """Parse week value."""
        if isinstance(val, pd.Timestamp):
            return val
        if isinstance(val, pd.Period):
            try:
                return val.end_time
            except Exception:
                return val.to_timestamp(how="end")
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return pd.NaT
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", s)
            if dates:
                return pd.to_datetime(dates[-1], errors="coerce")
            return pd.to_datetime(s, errors="coerce")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            s = str(int(val))
            if len(s) == 8:
                return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            return pd.to_datetime(val, errors="coerce")
        return pd.to_datetime(val, errors="coerce")
    
