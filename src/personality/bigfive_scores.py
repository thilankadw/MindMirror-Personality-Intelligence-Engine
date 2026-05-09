"""Utilities for bigfive scores."""
import logging
from abc import ABC, abstractmethod
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from time import perf_counter
import math
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PersonalityScorer(ABC):
    """Score personality data."""
    @abstractmethod
    def score(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Score the requested data."""
        pass

class BigFiveScorer(PersonalityScorer):
    
    """Score big five data."""
    def __init__(self, model_repo_id: str = "ppp57420/ocean-personality-distilbert", device: str = None):
        """Initialize the big five scorer."""
        self.model_repo_id = model_repo_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ocean_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        logger.info(f"BigFiveScorer: Initializing with model '{model_repo_id}' on device '{self.device}'")
        
        try:
            # Load model, tokenizer, and config
            self.tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
            self.config = AutoConfig.from_pretrained(model_repo_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_repo_id, 
                config=self.config
            ).eval()
            self.model = self.model.to(self.device)
            
            logger.info(f"BigFiveScorer: Model loaded successfully")
        except Exception as e:
            logger.error(f"BigFiveScorer: Failed to load model - {str(e)}")
            raise
    
    def _get_personality_scores(self, text: str) -> dict:
        """Get personality scores."""
        if not text or not text.strip():
            return {}
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions (regression output)
            with torch.no_grad():
                outputs = self.model(**inputs).logits.squeeze(0).cpu().tolist()
            
            # Convert to dictionary format
            # Order: openness, conscientiousness, extraversion, agreeableness, neuroticism
            scores = dict(zip(self.ocean_traits, outputs))
            
            return scores
        except Exception as e:
            logger.warning(f"BigFiveScorer: Error processing text - {str(e)}")
            return {}
    
    def _process_text_fields(self, row) -> str:
        """Process text fields."""
        title = row.get('title', '')
        selftext = row.get('selftext', '')
        
        # Handle NaN, None, or non-string values
        if not isinstance(title, str):
            title = '' if (title is None or (isinstance(title, float) and math.isnan(title))) else str(title)
        if not isinstance(selftext, str):
            selftext = '' if (selftext is None or (isinstance(selftext, float) and math.isnan(selftext))) else str(selftext)
        
        # Combine title and selftext
        text_concat = f"{title} {selftext}".strip()
        return text_concat if text_concat else ''
    
    def score(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Score the requested data."""
        total = len(posts)
        start = perf_counter()
        logger.info(f"BigFiveScorer: Starting personality scoring; posts={total}")
        
        try:
            posts = posts.copy()
            
            # Step 1: Create processed_text column
            logger.info("BigFiveScorer: Creating processed_text field...")
            posts['processed_text'] = posts.apply(self._process_text_fields, axis=1)
            
            # Check for empty texts
            empty_texts = (posts['processed_text'] == '').sum()
            empty_rate = empty_texts / total if total else 0.0
            logger.info(
                f"BigFiveScorer: Processed text created; empty={empty_texts}/{total} (rate={empty_rate:.3f})"
            )
            
            # Step 2: Generate scores for all posts
            logger.info("BigFiveScorer: Generating Big5 personality scores...")
            scores_list = []
            
            # Use tqdm for progress tracking
            for text in tqdm(posts['processed_text'], desc="Processing posts", disable=False):
                scores = self._get_personality_scores(text)
                scores_list.append(scores)
            
            # Add scores to dataframe
            posts['scores'] = scores_list
            posts['label_source'] = self.model_repo_id
            
            # Count posts with empty scores
            empty_scores = sum(1 for s in scores_list if len(s) == 0)
            empty_score_rate = empty_scores / total if total else 0.0
            
            duration_ms = (perf_counter() - start) * 1000
            logger.info(
                f"BigFiveScorer: Completed; duration_ms={duration_ms:.2f} "
                f"empty_scores={empty_scores}/{total} (rate={empty_score_rate:.3f})"
            )
            
            return posts
            
        except Exception as exc:
            logger.error(f"BigFiveScorer: Error during scoring - {str(exc)}")
            raise
    
    def expand_scores_to_columns(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Expand scores to columns."""
        try:
            logger.info("BigFiveScorer: Expanding scores to individual columns...")
            posts = posts.copy()
            
            # Extract scores into individual columns
            for trait in self.ocean_traits:
                posts[trait.capitalize()] = posts['scores'].apply(
                    lambda x: x.get(trait, np.nan) if isinstance(x, dict) else np.nan
                )
            
            logger.info(f"BigFiveScorer: Scores expanded to columns: {[t.capitalize() for t in self.ocean_traits]}")
            return posts
            
        except Exception as exc:
            logger.error(f"BigFiveScorer: Error expanding scores - {str(exc)}")
            raise
