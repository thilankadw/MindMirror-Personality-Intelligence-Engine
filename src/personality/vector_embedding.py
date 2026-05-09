"""Utilities for vector embedding."""
import json
from typing import Any, List
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class VectorEmbedding():
    """Generate vector embeddings."""
    def __init__(self):
        """Initialize the vector embedding."""
        try:
            from transformers import BertTokenizer, BertModel
            import torch
        except ImportError as exc:
            raise ImportError(
                "VectorEmbedding requires transformers and torch. "
                "Install them only if embedding generation is needed."
            ) from exc
        self._torch = torch
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        logger.info("BERT model and tokenizer loaded successfully")

    def get_bert_embedding(self, text):
        """Generate BERT embedding for a single text."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip():
            return np.zeros(768)
        with self._torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state[0, 0].numpy()  
        
    def build_embeddings(self, post_data: List[Dict[str, Any]]):
        """Build embeddings from post data structure."""
        user_posts = defaultdict(list)
        for entry in post_data:
            user = entry['user']
            for post in entry['posts']:
                text = post.get('preprocessed_text', post) if isinstance(post, dict) else post
                embedding = self.get_bert_embedding(text)
                user_posts[user].append(embedding)

        return user_posts
    
    def generate_embeddings(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_text',
        output_column: str = 'text_embeddings'
    ) -> pd.DataFrame:
        """Generate BERT embeddings for a DataFrame with text column."""
        try:
            logger.info(f"Generating embeddings for {len(df)} rows...")
            
            # Generate embeddings for each row
            embeddings = []
            for idx, text in enumerate(df[text_column]):
                if idx % 1000 == 0:
                    logger.info(f"Processing embedding {idx}/{len(df)}...")
                embedding = self.get_bert_embedding(text)
                embeddings.append(embedding)
            
            df[output_column] = embeddings
            logger.info(f"Embeddings generated. Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
