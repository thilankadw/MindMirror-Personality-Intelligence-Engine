"""Utilities for feature scaling."""
import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class FeatureScaler(ABC):
    """Provide feature scaler behavior."""
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Scale features in the dataframe and return the result."""
        pass

class RobustScale(FeatureScaler):
    """Provide robust scale behavior."""
    def __init__(self):
        """Initialize the robust scale."""
        self.scaler = RobustScaler()
        self.fitted = False

    def scale(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle scale."""
        try:
            logger.info(f"Scaling {len(columns)} columns using RobustScaler")
            df_copy = df.copy()
            df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
            logger.info("Scaling completed successfully")
            return df_copy
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def fit(self, X):
        """Fit the requested data."""
        try:
            logger.info("Fitting scaler on training data...")
            self.scaler.fit(X)
            self.fitted = True
            logger.info("Scaler fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
            raise
    
    def transform(self, X):
        """Transform the requested data."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        try:
            logger.info("Transforming data with fitted scaler...")
            transformed = self.scaler.transform(X)
            logger.info("Transform completed successfully")
            return transformed
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, X):
        """Fit and transform the input data."""
        self.fit(X)
        return self.transform(X)


    
