"""Utilities for model building."""
import os
from abc import ABC, abstractmethod
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModelBuilder(ABC):
    """Build base model."""
    def __init__(self, model_name: str, **kwargs):
        """Initialize the base model builder."""
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs

    @abstractmethod
    def build_model(self):
        """Build model."""
        pass

    def save_model(self, filepath):
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save. Build the model first")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model."""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

class RidgeModelBuilder(BaseModelBuilder):
    """Ridge Regression model builder for multi-output trait change prediction."""
    
    def __init__(self, feature_columns=None, **kwargs):
        """Initialize the ridge model builder."""
        default_params = {
            'alpha': 5.0,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('Ridge', **default_params)
        self.feature_columns = feature_columns

    def build_model(self):
        """Build model."""
        logger.info(f"Building Ridge model with params: {self.model_params}")
        
        if self.feature_columns is None:
            raise ValueError("feature_columns must be provided to build the model")
        
        self.model = Pipeline(steps=[
            ("reg", MultiOutputRegressor(Ridge(**self.model_params))),
        ])
        
        logger.info("Ridge model pipeline built successfully")
        return self.model


class GRUModelBuilder(BaseModelBuilder):
    """GRU (Gated Recurrent Unit) model builder for time series trait change prediction."""
    
    def __init__(self, timesteps=1, n_features=None, output_dim=5, **kwargs):
        """Initialize the gru model builder."""
        default_params = {
            'gru_units_1': 64,
            'gru_units_2': 32,
            'dropout_1': 0.3,
            'dropout_2': 0.2,
            'dense_units': 16,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 32,
            'early_stopping_patience': 15,
            'validation_split': 0.2,
            'verbose': 1
        }
        default_params.update(kwargs)
        super().__init__('GRU', **default_params)
        
        self.timesteps = timesteps
        self.n_features = n_features
        self.output_dim = output_dim
        self.history = None

    def build_model(self):
        """Build the GRU model architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for GRUModelBuilder. Install tensorflow-cpu to use this model."
            ) from exc

        if self.n_features is None:
            raise ValueError("n_features must be provided to build the GRU model")
        
        logger.info(f"Building GRU model with params: {self.model_params}")
        logger.info(f"Input shape: (timesteps={self.timesteps}, features={self.n_features})")
        logger.info(f"Output dimension: {self.output_dim}")
        
        self.model = keras.Sequential([
            layers.GRU(
                self.model_params['gru_units_1'], 
                return_sequences=True, 
                input_shape=(self.timesteps, self.n_features)
            ),
            layers.Dropout(self.model_params['dropout_1']),
            layers.GRU(
                self.model_params['gru_units_2'], 
                return_sequences=False
            ),
            layers.Dropout(self.model_params['dropout_2']),
            layers.Dense(self.model_params['dense_units'], activation='relu'),
            layers.Dense(self.output_dim)  # Linear output for regression
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.model_params['learning_rate']
            ),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("GRU model built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
        return self.model


class RandomForestModelBuilder(BaseModelBuilder):
    """Random Forest classifier builder for direction prediction."""

    def __init__(self, **kwargs):
        """Initialize the random forest model builder."""
        default_params = {
            "n_estimators": 400,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": None,
        }
        default_params.update(kwargs)
        super().__init__("RandomForest", **default_params)

    def build_model(self):
        """Build model."""
        logger.info("Building RandomForestClassifier with params: %s", self.model_params)
        self.model = RandomForestClassifier(**self.model_params)
        return self.model


class HistGradientBoostingModelBuilder(BaseModelBuilder):
    """Histogram-based Gradient Boosting classifier builder."""

    def __init__(self, **kwargs):
        """Initialize the hist gradient boosting model builder."""
        default_params = {
            "random_state": 42,
            "max_depth": 6,
            "learning_rate": 0.1,
        }
        default_params.update(kwargs)
        super().__init__("HistGradientBoosting", **default_params)

    def build_model(self):
        """Build model."""
        logger.info(
            "Building HistGradientBoostingClassifier with params: %s",
            self.model_params,
        )
        self.model = HistGradientBoostingClassifier(**self.model_params)
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    """XGBoost classifier builder (requires xgboost)."""

    def __init__(self, **kwargs):
        """Initialize the xg boost model builder."""
        default_params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        super().__init__("XGBoost", **default_params)

    def build_model(self):
        """Build model."""
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for XGBoostModelBuilder. Install with `pip install xgboost`."
            )
        logger.info("Building XGBClassifier with params: %s", self.model_params)
        self.model = XGBClassifier(**self.model_params)
        return self.model
