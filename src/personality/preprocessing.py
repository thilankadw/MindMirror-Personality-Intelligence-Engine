"""Utilities for preprocessing."""
import logging
from typing import Optional

import pandas as pd

from src.personality.data_parser import TraitScoreParser
from src.personality.data_preprocess import (
    AttributeConcatenation,
    CleanText,
    DropAttributes,
    ReplaceEmoji,
)
from src.personality.delta_targets import StdNaNHandler
from src.personality.weekly_preprocess import GapWeekCalculator, TraitWeeklyAggregator
from utils.config import (
    get_created_date_column,
    get_grouping_feature,
    get_required_attributes,
    get_scores_column,
    get_selftext_column,
    get_std_features,
    get_time_sort_column,
    get_title_column,
    get_traits,
    get_week_column,
)

logger = logging.getLogger(__name__)


class PersonalityInferencePreprocessor:
    """Shared preprocessing flow for all Big Five trait inference pipelines."""

    def __init__(self, domain: str = "personality"):
        """Initialize the personality inference preprocessor."""
        self.domain = domain

        grouping_features = get_grouping_feature(domain=domain)
        if len(grouping_features) != 1:
            raise ValueError(
                "Expected exactly one grouping feature in config.yaml: "
                "domains.<domain>.features.group_feature"
            )

        self.author_col = grouping_features[0]
        self.week_col = get_week_column(domain=domain)
        self.time_sort_col = get_time_sort_column(domain=domain)
        self.created_date_col = get_created_date_column(domain=domain)
        self.scores_col = get_scores_column(domain=domain)
        self.title_col = get_title_column(domain=domain)
        self.selftext_col = get_selftext_column(domain=domain)
        self.traits = get_traits(domain=domain)
        self.std_features = get_std_features(domain=domain)

        required_attributes = get_required_attributes(domain=domain)
        self.attribute_concat = AttributeConcatenation(
            title_col=self.title_col,
            selftext_col=self.selftext_col,
        )
        self.drop_attributes = DropAttributes(required_attributes)
        self.text_clean = CleanText()
        self.replace_emoji = ReplaceEmoji()
        self.trait_parser = TraitScoreParser(scores_column=self.scores_col)
        self.weekly_aggregator = TraitWeeklyAggregator(
            trait_columns=self.traits,
            created_date_col=self.created_date_col,
            week_col=self.week_col,
        )
        self.gap_calculator = GapWeekCalculator(
            author_col=self.author_col,
            week_col=self.week_col,
            time_col=self.time_sort_col,
        )
        self.nan_handler = StdNaNHandler(
            trait_columns=self.traits,
            std_cols=self.std_features,
        )

    def prepare(
        self,
        posts_df: pd.DataFrame,
        author: Optional[str] = None,
        gap_weeks: float = 0.0,
    ) -> pd.DataFrame:
        """Prepare the requested data."""
        df = posts_df.copy()
        self._validate_and_enrich_input(df=df, author=author)

        logger.info("Processing posts through preprocessing pipeline...")
        logger.info("Input shape: %s", df.shape)

        logger.info("Step 1/8: Concatenating title and selftext...")
        df = self.attribute_concat.preprocess(df)

        logger.info("Step 2/8: Dropping unused attributes...")
        df = self.drop_attributes.preprocess(df)

        logger.info("Step 3/8: Cleaning text...")
        df = self.text_clean.preprocess(df)

        logger.info("Step 4/8: Replacing emojis...")
        df = self.replace_emoji.preprocess(df)

        logger.info("Step 5/8: Parsing trait scores...")
        df = self.trait_parser.parse(df)

        logger.info("Step 6/8: Aggregating by week...")
        df = self.weekly_aggregator.aggregate(df, groupby_cols=[self.author_col, self.week_col])

        if len(df) > 1:
            logger.info("Step 7/8: Calculating gap weeks...")
            df = self.gap_calculator.calculate(df)
        else:
            logger.info("Step 7/8: Single week detected, setting gap_weeks=%s", gap_weeks)
            df["gap_weeks"] = gap_weeks

        logger.info("Step 8/8: Handling NaN values...")
        df = self.nan_handler.handle(df, groupby_col=self.author_col)

        self._fill_remaining_std_nans(df)

        if "gap_weeks" not in df.columns:
            logger.info("Adding gap_weeks=%s for inference", gap_weeks)
            df["gap_weeks"] = gap_weeks

        logger.info("Preprocessing complete. Output shape: %s", df.shape)
        return df

    def _validate_and_enrich_input(
        self,
        df: pd.DataFrame,
        author: Optional[str] = None,
    ) -> None:
        """Validate and enrich input."""
        if self.author_col not in df.columns:
            if author is None:
                raise ValueError(
                    f"Either '{self.author_col}' column in data or 'author' parameter is required"
                )
            df[self.author_col] = author

        basic_cols = [self.author_col, self.created_date_col]
        if self.scores_col not in df.columns:
            missing_text = [
                col for col in [self.title_col, self.selftext_col] if col not in df.columns
            ]
            if missing_text:
                raise ValueError(f"Input data missing required text columns: {missing_text}")

        missing = [col for col in basic_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")

    @staticmethod
    def _fill_remaining_std_nans(df: pd.DataFrame) -> None:
        """Fill remaining std nans."""
        std_cols = [col for col in df.columns if col.endswith("_std_w")]
        for col in std_cols:
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                logger.info(
                    "Filling %s remaining NaN in %s with 0.0 (insufficient variation)",
                    nan_count,
                    col,
                )
                df[col] = df[col].fillna(0.0)
