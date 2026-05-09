"""Utilities for data parser."""
import logging
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
import ast
import json
import re

logger = logging.getLogger(__name__)

_BIG5_TRAITS = {
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
}


class DataParser(ABC):
    """Abstract base class for parsing data."""
    
    @abstractmethod
    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and prepare data."""
        pass

class TraitScoreParser(DataParser):
    """Parse trait scores and expand them into separate columns."""

    def __init__(self, scores_column: str = "scores"):
        """Initialize the trait score parser."""
        self.scores_column = scores_column

    def _normalize_trait_columns(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """Normalize parsed Big5 trait columns to canonical lowercase names."""
        rename_map = {}
        for col in df_traits.columns:
            if not isinstance(col, str):
                continue
            normalized = col.strip().lower()
            if normalized in _BIG5_TRAITS and normalized not in df_traits.columns and col != normalized:
                rename_map[col] = normalized

        if rename_map:
            logger.info("Normalizing parsed trait columns: %s", rename_map)
            df_traits = df_traits.rename(columns=rename_map)
        return df_traits

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the requested data."""
        try:
            logger.info("Parsing trait scores...")
            df_copy = df.copy()

            if self.scores_column not in df_copy.columns:
                raise ValueError(f"Missing scores column: {self.scores_column}")

            if isinstance(df_copy[self.scores_column].iloc[0], str):
                df_copy[self.scores_column] = df_copy[self.scores_column].apply(ast.literal_eval)
            
            df_traits = pd.json_normalize(df_copy[self.scores_column])
            df_traits = self._normalize_trait_columns(df_traits)
            df_result = pd.concat([df_copy, df_traits], axis=1)

            df_result = df_result.drop(columns=[self.scores_column])
            
            logger.info(f"Trait scores parsed. Columns: {df_result.columns.tolist()}")
            return df_result
            
        except Exception as e:
            logger.error(f"Error parsing trait scores: {str(e)}")
            raise


class TextEmbeddingParser(DataParser):
    """Parse text embeddings from string format to numpy arrays."""

    def __init__(
        self,
        embedding_column: str = "text_embeddings",
        output_column: str = None,
        validate_dims: bool = True,
    ):
        """Initialize the text embedding parser."""
        self.embedding_column = embedding_column
        self.output_column = output_column
        self.validate_dims = validate_dims

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the requested data."""
        try:
            logger.info("Parsing text embeddings...")
            df_copy = df.copy()

            emb_col = self.embedding_column
            if emb_col not in df_copy.columns:
                raise ValueError(f"Missing embedding column: {emb_col}")
            out_col = self.output_column or emb_col
            logger.info("Embedding columns: source=%s output=%s", emb_col, out_col)

            df_copy[out_col] = df_copy[emb_col].apply(self._parse_embedding)

            if self.validate_dims:
                self._validate_embedding_dims(df_copy, emb_col, out_col)

            null_count = int(df_copy[out_col].isna().sum())
            logger.info("Parsed embeddings: rows=%s nulls=%s", len(df_copy), null_count)
            logger.info("Text embeddings parsed successfully.")
            return df_copy

        except Exception as e:
            logger.error(f"Error parsing text embeddings: {str(e)}")
            raise

    def _is_nan(self, val) -> bool:
        """Return whether nan."""
        return val is None or (isinstance(val, float) and np.isnan(val))

    def _parse_numeric_tokens(self, text: str):
        """Parse numeric tokens."""
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not tokens:
            return None
        return np.array([float(t) for t in tokens], dtype=float)

    def _parse_embedding(self, value):
        """Convert embedding input to numpy array or NaN."""
        if isinstance(value, np.ndarray):
            return value.astype(float)
        if isinstance(value, list):
            return np.array(value, dtype=float)
        if self._is_nan(value):
            return np.nan
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                return np.nan

            low = s.lower()
            if low.startswith("array(") or low.startswith("tensor("):
                left = s.find("(")
                right = s.rfind(")")
                if left != -1 and right != -1 and right > left:
                    s = s[left + 1:right].strip()

            try:
                if s.startswith("["):
                    return np.array(ast.literal_eval(s), dtype=float)
                m = re.search(r"\[.*\]", s)
                if m:
                    return np.array(ast.literal_eval(m.group(0)), dtype=float)
                return np.array(ast.literal_eval(s), dtype=float)
            except Exception:
                try:
                    return np.array(json.loads(s), dtype=float)
                except Exception:
                    parsed = self._parse_numeric_tokens(s)
                    if parsed is not None:
                        return parsed
                    return np.nan

        try:
            return np.array(value, dtype=float)
        except Exception:
            return np.nan

    def _validate_embedding_dims(self, df: pd.DataFrame, raw_col: str, parsed_col: str):
        """Validate embedding dims."""
        emb_lens = df[parsed_col].dropna().apply(
            lambda v: int(len(v)) if hasattr(v, "__len__") else np.nan
        )
        if emb_lens.dropna().empty:
            sample_raw = df[raw_col].dropna().astype(str).head(3).tolist()
            raise ValueError(
                f"No valid embeddings parsed. Example raw values: {sample_raw}"
            )

        unique_dims = sorted({int(v) for v in emb_lens.dropna().unique()})
        if len(unique_dims) != 1:
            raise ValueError(f"Embedding dimensionality mismatch: {unique_dims}")

        logger.info("Parsed embedding dimension: %s", unique_dims[0])
