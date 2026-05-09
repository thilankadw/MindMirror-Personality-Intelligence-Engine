"""Utilities for inference feature builder."""
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.personality.data_parser import TextEmbeddingParser, TraitScoreParser
from src.personality.data_preprocess import (
    AttributeConcatenation,
    CleanText,
    DropAttributes,
    ReplaceEmoji,
)
from src.personality.delta_targets import StdNaNHandler
from src.personality.lagged_features import FeatureShiftCreator, RollingStatsFeatureCreator
from src.personality.weekly_preprocess import (
    EmbeddingWeeklyAggregator,
    GapWeekCalculator,
    TraitWeeklyAggregator,
)
from utils.config import (
    get_created_date_column,
    get_data_paths,
    get_grouping_feature,
    get_processed_text_column,
    get_raw_embedding_column,
    get_required_attributes,
    get_scores_column,
    get_selftext_column,
    get_std_features,
    get_text_embedding_feature,
    get_time_sort_column,
    get_title_column,
    get_traits,
    get_week_column,
)

logger = logging.getLogger(__name__)


class PersonalityInferenceFeatureBuilder:
    """Build model-ready inference features directly from raw Reddit posts."""

    def __init__(self, trait: str, domain: str = "personality"):
        """Initialize the personality inference feature builder."""
        if not trait:
            raise ValueError("trait is required")

        grouping_features = get_grouping_feature(domain=domain)
        if len(grouping_features) != 1:
            raise ValueError(
                "Expected exactly one grouping feature in config.yaml: "
                "domains.<domain>.features.group_feature"
            )

        self.domain = domain
        self.trait = trait
        self.author_col = grouping_features[0]
        self.week_col = get_week_column(domain=domain)
        self.time_sort_col = get_time_sort_column(domain=domain)
        self.created_date_col = get_created_date_column(domain=domain)
        self.processed_text_col = get_processed_text_column(domain=domain)
        self.raw_embedding_col = get_raw_embedding_column(domain=domain)
        self.scores_col = get_scores_column(domain=domain)
        self.title_col = get_title_column(domain=domain)
        self.selftext_col = get_selftext_column(domain=domain)
        self.traits = get_traits(domain=domain)
        self.embedding_feature_col = get_text_embedding_feature(domain=domain)[0]
        self.std_col = get_std_features(domain=domain, trait=trait)[0]
        self.obs_col = f"{trait}_mean_w"
        self.lag_col = f"{trait}_lag1"
        self.roll_mean_col = f"{trait}_roll4"
        self.roll_std_col = f"{trait}_roll4_std"

        self.attribute_concat = AttributeConcatenation(
            title_col=self.title_col,
            selftext_col=self.selftext_col,
        )
        self.text_clean = CleanText()
        self.replace_emoji = ReplaceEmoji()
        self.drop_attributes = DropAttributes(get_required_attributes(domain=domain))
        self.trait_parser = TraitScoreParser(scores_column=self.scores_col)
        self.embedding_parser = TextEmbeddingParser(
            embedding_column=self.raw_embedding_col,
            output_column=self.raw_embedding_col,
            validate_dims=True,
        )
        self.weekly_trait_aggregator = TraitWeeklyAggregator(
            trait_columns=[trait],
            created_date_col=self.created_date_col,
            week_col=self.week_col,
        )
        self.weekly_embedding_aggregator = EmbeddingWeeklyAggregator(
            embedding_column=self.raw_embedding_col,
            output_column=self.embedding_feature_col,
            created_date_col=self.created_date_col,
            week_col=self.week_col,
        )
        self.gap_calculator = GapWeekCalculator(
            author_col=self.author_col,
            week_col=self.week_col,
            time_col=self.time_sort_col,
        )
        self.nan_handler = StdNaNHandler(
            trait_columns=[trait],
            std_cols=[self.std_col],
        )

        self._scorer = None
        self._embedder = None

    def prepare(
        self,
        posts_df: pd.DataFrame,
        author: Optional[str] = None,
        gap_weeks: float = 0.0,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare the requested data."""
        df = self._normalize_input(posts_df.copy(), author=author)
        logger.info("Preparing raw posts for %s inference. Input shape: %s", self.trait, df.shape)

        df = self._ensure_processed_text(df)
        df = self._ensure_scores(df)
        df = self._ensure_trait_columns(df)
        df = self._ensure_embeddings(df)
        df = self._slim_columns(df)

        weekly_df = self._build_weekly_frame(df, gap_weeks=gap_weeks)
        feature_df, feature_columns = self._build_model_features(weekly_df)
        logger.info(
            "Feature preparation complete. Output shape: %s feature_count=%s",
            feature_df.shape,
            len(feature_columns),
        )
        return feature_df, feature_columns

    def _normalize_input(self, df: pd.DataFrame, author: Optional[str]) -> pd.DataFrame:
        """Normalize input."""
        if self.author_col not in df.columns:
            for fallback_col in [
                "provider_username",
                "platform_username",
                "external_username",
                "username",
                "user",
            ]:
                if fallback_col in df.columns:
                    df[self.author_col] = df[fallback_col]
                    break

        if self.author_col not in df.columns:
            if author is None:
                raise ValueError(
                    f"Either '{self.author_col}' column in data or 'author' parameter is required"
                )
            df[self.author_col] = author
        elif author is not None:
            df[self.author_col] = df[self.author_col].fillna(author)

        if self.created_date_col not in df.columns:
            if "created_at" in df.columns:
                df[self.created_date_col] = df["created_at"]
            elif "created_utc" in df.columns:
                df[self.created_date_col] = pd.to_datetime(
                    df["created_utc"],
                    unit="s",
                    utc=True,
                    errors="coerce",
                )

        if self.created_date_col not in df.columns:
            raise ValueError(
                f"Input data missing required timestamp columns: '{self.created_date_col}', "
                "'created_at', or 'created_utc'"
            )

        for col in [self.title_col, self.selftext_col]:
            if col not in df.columns:
                df[col] = ""

        return df

    def _ensure_processed_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure processed text."""
        if self.processed_text_col not in df.columns:
            df = self.attribute_concat.preprocess(df)
        df = self.text_clean.preprocess(df)
        df = self.replace_emoji.preprocess(df)
        return df

    def _ensure_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure scores."""
        has_required_trait_cols = self.trait in df.columns or all(
            trait in df.columns for trait in self.traits
        )
        if self.scores_col in df.columns or has_required_trait_cols:
            return df

        logger.info("Generating Big Five scores for %s posts...", len(df))
        scorer = self._get_scorer()
        df = df.copy()
        df[self.scores_col] = self._score_texts(
            texts=df[self.processed_text_col].tolist(),
            scorer=scorer,
        )
        df["label_source"] = scorer.model_repo_id
        return df

    def _ensure_trait_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure trait columns."""
        if self.trait in df.columns or all(trait in df.columns for trait in self.traits):
            return df
        if self.scores_col not in df.columns:
            raise ValueError(f"Missing '{self.scores_col}' needed to derive trait columns.")
        return self.trait_parser.parse(df)

    def _ensure_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure embeddings."""
        if self.raw_embedding_col not in df.columns:
            logger.info("Generating BERT embeddings for %s posts...", len(df))
            df = df.copy()
            df[self.raw_embedding_col] = self._embed_texts(
                texts=df[self.processed_text_col].tolist(),
                embedder=self._get_embedder(),
            )
        return self.embedding_parser.parse(df)

    def _slim_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle slim columns."""
        if self.scores_col not in df.columns:
            return df
        slim_df = self.drop_attributes.preprocess(df)
        for trait in self.traits:
            if trait in df.columns and trait not in slim_df.columns:
                slim_df[trait] = df[trait].values
        return slim_df

    def _build_weekly_frame(self, df: pd.DataFrame, gap_weeks: float) -> pd.DataFrame:
        """Build weekly frame."""
        weekly_traits = self.weekly_trait_aggregator.aggregate(
            df,
            groupby_cols=[self.author_col, self.week_col],
        )
        weekly_embeddings = self.weekly_embedding_aggregator.aggregate(
            df,
            groupby_cols=[self.author_col, self.week_col],
        )

        weekly_df = weekly_traits.merge(
            weekly_embeddings,
            on=[self.author_col, self.week_col],
            how="inner",
        )
        weekly_df = self.nan_handler.handle(weekly_df, groupby_col=self.author_col)
        weekly_df[self.std_col] = weekly_df[self.std_col].fillna(0.0)

        if self.time_sort_col not in weekly_df.columns:
            weekly_df[self.time_sort_col] = weekly_df[self.week_col].apply(
                self.gap_calculator._parse_week_value
            )

        if len(weekly_df) > 1:
            weekly_df = self.gap_calculator.calculate(weekly_df)
        else:
            weekly_df = weekly_df.copy()
            weekly_df["time_gap_weeks"] = gap_weeks
            weekly_df["log_time_gap_weeks"] = np.log1p(max(gap_weeks, 0.0))
            weekly_df["gap_weeks"] = gap_weeks

        return weekly_df.sort_values([self.author_col, self.time_sort_col]).reset_index(drop=True)

    def _build_model_features(self, weekly_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Build model features."""
        df = weekly_df.copy()

        df[self.lag_col] = df[self.obs_col]
        df = RollingStatsFeatureCreator(
            value_column=self.obs_col,
            group_column=self.author_col,
            window=4,
            min_periods=2,
            mean_column=self.roll_mean_col,
            std_column=self.roll_std_col,
        ).create(df)

        df = FeatureShiftCreator(
            feature_columns=[self.lag_col, self.roll_mean_col, self.roll_std_col],
            group_column=self.author_col,
            lag=1,
        ).create(df)
        df["emb_vec"] = df[self.embedding_feature_col].apply(lambda value: np.asarray(value, dtype=float))
        df["emb_vec_shift"] = df.groupby(self.author_col)["emb_vec"].shift(1)

        numeric_features = [
            "time_gap_weeks",
            "log_time_gap_weeks",
            self.lag_col,
            self.roll_mean_col,
            self.roll_std_col,
            self.std_col,
        ]
        required = numeric_features + ["emb_vec_shift"]
        model_df = df.dropna(subset=required).copy()

        if model_df.empty:
            raise ValueError(
                "No inference rows remain after lag/shift feature construction. "
                "At least two weekly observations are required."
            )

        emb_matrix = np.vstack(
            model_df["emb_vec_shift"].apply(lambda value: np.asarray(value, dtype=float)).to_numpy()
        )
        emb_feature_cols = [f"emb_vec_shift_{index}" for index in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, columns=emb_feature_cols, index=model_df.index)
        model_df = pd.concat([model_df, emb_df], axis=1)

        feature_columns = self._resolve_feature_columns(emb_feature_cols)
        missing_cols = [col for col in feature_columns if col not in model_df.columns]
        if missing_cols:
            raise ValueError(f"Constructed inference frame is missing model columns: {missing_cols}")

        return model_df, feature_columns

    def _resolve_feature_columns(self, emb_feature_cols: List[str]) -> List[str]:
        """Resolve feature columns."""
        metadata_feature_cols = self._load_feature_columns_from_metadata()
        if metadata_feature_cols:
            expected = set(metadata_feature_cols)
            actual = set(
                [
                    "time_gap_weeks",
                    "log_time_gap_weeks",
                    self.lag_col,
                    self.roll_mean_col,
                    self.roll_std_col,
                    self.std_col,
                ]
                + emb_feature_cols
            )
            if expected.issubset(actual):
                return metadata_feature_cols
            logger.warning(
                "Metadata feature schema does not match constructed columns for %s. Falling back.",
                self.trait,
            )

        return [
            "time_gap_weeks",
            "log_time_gap_weeks",
            self.lag_col,
            self.roll_mean_col,
            self.roll_std_col,
            self.std_col,
        ] + emb_feature_cols

    def _load_feature_columns_from_metadata(self) -> List[str]:
        """Load feature columns from metadata."""
        metadata_path = get_data_paths(domain=self.domain, trait=self.trait).get(
            "data_pipeline_metadata"
        )
        if not metadata_path or not os.path.exists(metadata_path):
            return []

        try:
            with open(metadata_path, "r", encoding="utf-8") as file_handle:
                metadata = json.load(file_handle)
            feature_names = metadata.get("feature_names")
            if isinstance(feature_names, list):
                return [str(name) for name in feature_names]
        except Exception as exc:
            logger.warning("Failed to load feature metadata for %s: %s", self.trait, exc)
        return []

    def _get_scorer(self):
        """Get scorer."""
        if self._scorer is None:
            from src.personality.bigfive_scores import BigFiveScorer

            self._scorer = BigFiveScorer()
        return self._scorer

    def _get_embedder(self):
        """Get embedder."""
        if self._embedder is None:
            from src.personality.vector_embedding import VectorEmbedding

            self._embedder = VectorEmbedding()
        return self._embedder

    def _score_texts(self, texts: List[object], scorer, batch_size: int = 16) -> List[dict]:
        """Score texts."""
        import torch

        results: List[dict] = []
        cleaned_texts = [self._normalize_text(text) for text in texts]

        for start in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[start : start + batch_size]
            valid_items = [(index, text) for index, text in enumerate(batch) if text]
            batch_scores = [{} for _ in batch]

            if valid_items:
                inputs = scorer.tokenizer(
                    [text for _, text in valid_items],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                inputs = {key: value.to(scorer.device) for key, value in inputs.items()}
                with torch.no_grad():
                    logits = scorer.model(**inputs).logits.detach().cpu().numpy()

                for offset, values in zip([idx for idx, _ in valid_items], logits):
                    batch_scores[offset] = dict(zip(scorer.ocean_traits, values.tolist()))

            results.extend(batch_scores)

        return results

    def _embed_texts(self, texts: List[object], embedder, batch_size: int = 16) -> List[np.ndarray]:
        """Handle embed texts."""
        results: List[np.ndarray] = []
        cleaned_texts = [self._normalize_text(text) for text in texts]
        zero_embedding = np.zeros(768, dtype=float)

        for start in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[start : start + batch_size]
            valid_items = [(index, text) for index, text in enumerate(batch) if text]
            batch_embeddings: List[np.ndarray] = [zero_embedding.copy() for _ in batch]

            if valid_items:
                inputs = embedder.tokenizer(
                    [text for _, text in valid_items],
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                with embedder._torch.no_grad():
                    outputs = embedder.bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

                for offset, values in zip([idx for idx, _ in valid_items], cls_embeddings):
                    batch_embeddings[offset] = values.astype(float)

            results.extend(batch_embeddings)

        return results

    @staticmethod
    def _normalize_text(text: object) -> str:
        """Normalize text."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
