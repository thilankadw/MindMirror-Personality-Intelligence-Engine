"""Utilities for data ingestion."""
import os
import json
import logging
import ast
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

_BIG5_TITLE_TO_LOWER = {
    "Openness": "openness",
    "Conscientiousness": "conscientiousness",
    "Extraversion": "extraversion",
    "Agreeableness": "agreeableness",
    "Neuroticism": "neuroticism",
}


def _normalize_personality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known Big5 column aliases to canonical lowercase names."""
    rename_map: Dict[str, str] = {}

    for src, dst in _BIG5_TITLE_TO_LOWER.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

        src_mean = f"{src}_mean_w"
        dst_mean = f"{dst}_mean_w"
        if src_mean in df.columns and dst_mean not in df.columns:
            rename_map[src_mean] = dst_mean

        src_std = f"{src}_std_w"
        dst_std = f"{dst}_std_w"
        if src_std in df.columns and dst_std not in df.columns:
            rename_map[src_std] = dst_std

    if rename_map:
        logger.info("Normalizing personality column names: %s", rename_map)
        df = df.rename(columns=rename_map)

    return df


class DataIngestor(ABC):
    """Provide data ingestor behavior."""
    @abstractmethod
    def ingest(self, file_path_or_links: str) -> List[Dict[str, Any]]:
        """Ingest the requested data."""
        pass

class DataIngestorJson(DataIngestor):
    """Provide data ingestor JSON behavior."""
    def ingest(self, file_path_or_link):
        """Ingest the requested data."""
        try:
            logger.info("Ingesting JSON data from: %s", file_path_or_link)
            with open(file_path_or_link, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                data_df = pd.DataFrame(data)
                data_df = _normalize_personality_columns(data_df)
                data = data_df.to_dict(orient="records")
            logger.info("Successfully ingested data from %s (items=%s)", file_path_or_link,
                        (len(data) if hasattr(data, '__len__') else type(data)))
            return data
        except Exception as e:
            logger.exception("Failed to ingest JSON data from %s: %s", file_path_or_link, e)
            raise

class DataIngestorCSV(DataIngestor):
    """Provide data ingestor CSV behavior."""
    def __init__(
        self,
        embedding_columns: Optional[List[str]] = None,
        return_records: bool = False,
        usecols: Optional[List[str]] = None,
    ):
        """Initialize the data ingestor CSV."""
        self.embedding_columns = (
            ["text_embeddings"] if embedding_columns is None else embedding_columns
        )
        self.return_records = return_records
        self.usecols = usecols

    @staticmethod
    def get_column_map(file_path_or_link: str) -> Dict[str, str]:
        """Map canonical column names back to the actual CSV header names."""
        header_df = pd.read_csv(file_path_or_link, nrows=0)
        source_columns = header_df.columns.tolist()
        normalized_columns = _normalize_personality_columns(header_df).columns.tolist()
        return dict(zip(normalized_columns, source_columns))

    @staticmethod
    def get_columns(file_path_or_link: str) -> List[str]:
        """Read only the CSV header so callers can choose a minimal column set."""
        return list(DataIngestorCSV.get_column_map(file_path_or_link).keys())

    def ingest(self, file_path_or_link):
        """Ingest the requested data."""
        try:
            logger.info("Ingesting CSV data from: %s", file_path_or_link)
            if not os.path.exists(file_path_or_link):
                raise FileNotFoundError(f"CSV file not found: {file_path_or_link}")

            if self.usecols:
                logger.info("Loading CSV with restricted columns: %s", self.usecols)
            df = pd.read_csv(file_path_or_link, usecols=self.usecols)
            df = _normalize_personality_columns(df)
            logger.info("CSV loaded. Shape: %s", df.shape)

            for col in self.embedding_columns:
                if col in df.columns:
                    logger.info("Parsing embedding column: %s", col)
                    df[col] = df[col].apply(self._parse_embedding)
                    nulls = int(df[col].isna().sum())
                    logger.info("Embedding column parsed: %s nulls=%s", col, nulls)

            data = df.to_dict(orient="records")
            logger.info(
                "Successfully ingested data from %s (items=%s)",
                file_path_or_link,
                len(data),
            )
            return data if self.return_records else df

        except Exception as e:
            logger.exception("Failed to ingest CSV data from %s: %s", file_path_or_link, e)
            raise

    def _parse_embedding(self, value):
        """Parse embedding."""
        if isinstance(value, np.ndarray):
            return value.astype(float)
        if isinstance(value, list):
            return np.array(value, dtype=float)
        if value is None or (isinstance(value, float) and np.isnan(value)):
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
                m = re.search(r"\[.*\]", s, flags=re.S)
                if m:
                    return np.array(ast.literal_eval(m.group(0)), dtype=float)
                return np.array(ast.literal_eval(s), dtype=float)
            except Exception:
                try:
                    return np.array(json.loads(s), dtype=float)
                except Exception:
                    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                    if tokens:
                        return np.array([float(t) for t in tokens], dtype=float)
                    return np.nan

        try:
            return np.array(value, dtype=float)
        except Exception:
            return np.nan


DataIngestorCsv = DataIngestorCSV
