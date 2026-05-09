"""Utilities for data preprocess."""
import logging
from abc import ABC, abstractmethod
import pandas as pd
import math
import re
from time import perf_counter
import demoji

logger = logging.getLogger(__name__)


class DataPreprocessor(ABC):
    """Preprocess data data."""
    @abstractmethod
    def preprocess(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Transform a list of post dicts and return the result."""
        pass

class AttributeConcatenation(DataPreprocessor):
    """Provide attribute concatenation behavior."""
    def __init__(self, title_col: str = "title", selftext_col: str = "selftext"):
        """Initialize the attribute concatenation."""
        self.title_col = title_col
        self.selftext_col = selftext_col

    def process_text_fields(self, row):
            """Process text fields."""
            title = row[self.title_col]
            selftext = row[self.selftext_col]
            
            if not isinstance(title, str):
                title = '' if (title is None or (isinstance(title, float) and math.isnan(title))) else str(title)
            if not isinstance(selftext, str):
                selftext = '' if (selftext is None or (isinstance(selftext, float) and math.isnan(selftext))) else str(selftext)
            
            text_concat = f"{title} {selftext}".strip()
            return text_concat if text_concat else ''
    
    def preprocess(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the requested data."""
        total = len(posts)
        start = perf_counter()
        logger.info("AttributeConcatenation: start; posts=%d", total)

        try:
            missing = [
                col
                for col in [self.title_col, self.selftext_col]
                if col not in posts.columns
            ]
            if missing:
                raise ValueError(f"Missing required text columns: {missing}")
            posts = posts.copy()
            posts['processed_text'] = posts.apply(self.process_text_fields, axis=1)
            emptied = (posts['processed_text'] == '').sum()
            empty_rate = emptied / total if total else 0.0
            
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                "AttributeConcatenation: done; duration_ms=%.2f emptied=%d empty_rate=%.3f",
                duration_ms,
                emptied,
                empty_rate,
            )
        except Exception as exc:
            logger.error("AttributeConcatenation: error: %s", exc)
            raise

        return posts
    
class DropAttributes(DataPreprocessor):
    """Provide drop attributes behavior."""
    def __init__(self, req_attributes):
        """Initialize the drop attributes."""
        self.req_attributes = req_attributes

    def clean_media_summary(self, media_summary):
            """Clean media summary."""
            if media_summary is not None and isinstance(media_summary, list):
                for summary in media_summary:
                    if isinstance(summary, dict):
                        if 'file' in summary:
                            del summary['file']
                        if 'type' in summary:
                            del summary['type']
            return media_summary

    def preprocess(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the requested data."""
        total = len(posts)
        start = perf_counter()
        logger.info("DropAttributes: start; posts=%d keep=%s", total, self.req_attributes)
        if not self.req_attributes:
            logger.warning("DropAttributes: req_attributes is empty; resulting DataFrame will be empty")
        try:
            # Clean media_summary if it exists
            if 'media_summary' in posts.columns:
                posts['media_summary'] = posts['media_summary'].apply(self.clean_media_summary)
            
            # Keep only required columns
            available_attrs = [attr for attr in self.req_attributes if attr in posts.columns]
            missing_attrs = [attr for attr in self.req_attributes if attr not in posts.columns]
            
            if missing_attrs:
                logger.warning("DropAttributes: missing columns=%s", missing_attrs)
            
            posts = posts[available_attrs]
            
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                "DropAttributes: done; duration_ms=%.2f",
                duration_ms,
            )
        except Exception as exc:
            logger.error("DropAttributes: error: %s", exc)
            raise

        return posts

class CleanText(DataPreprocessor):
    """Provide clean text behavior."""
    def __init__(self):
        """Initialize the clean text."""
        self.URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self.EMPTY_PARENS_PATTERN = re.compile(r'\(\s*\)')
        self.NEWLINE_PATTERN = re.compile(r'[\r\n]+')
    
    def clean_text(self, text):
        """Clean text."""
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = ''
        elif not isinstance(text, str):
            text = str(text)
        
        text = self.MARKDOWN_LINK_PATTERN.sub(r'\1', text)
        text = self.URL_PATTERN.sub('[URL]', text)
        text = self.NEWLINE_PATTERN.sub(' ', text)
        text = self.EMPTY_PARENS_PATTERN.sub('', text)
        
        return text.strip()
    
    def preprocess(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the requested data."""
        total = len(posts)
        start = perf_counter()
        logger.info("CleanText: start; posts=%d", total)

        try:
            posts = posts.copy()
            posts['processed_text'] = posts['processed_text'].apply(self.clean_text)
            
            duration_ms = (perf_counter() - start) * 1000
            logger.info("CleanText: done; duration_ms=%.2f", duration_ms)
        except Exception as exc:
            logger.error("CleanText: error: %s", exc)
            raise

        return posts

class ReplaceEmoji(DataPreprocessor):
    """Provide replace emoji behavior."""
    def replace_emoji(self, text: str) -> str:
        """Replace emoji."""
        if not text:
            return text
        if not isinstance(text, str):
            text = str(text)
        
        emojis = demoji.findall(text)
        for emoji_char, description in emojis.items():
            normalized = str(description).replace(" ", "_")
            text = text.replace(emoji_char, f"[emoji_{normalized}]")
        return text

    def preprocess(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the requested data."""
        total = len(posts)
        start = perf_counter()
        logger.info("ReplaceEmoji: start; posts=%d", total)
        
        try:
            posts = posts.copy()
            posts['processed_text'] = posts['processed_text'].apply(self.replace_emoji)
            
            duration_ms = (perf_counter() - start) * 1000
            logger.info("ReplaceEmoji: done; duration_ms=%.2f", duration_ms)
        except Exception as exc:
            logger.error("ReplaceEmoji: error: %s", exc)
            raise

        return posts
