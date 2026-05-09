"""Utilities for model training."""
import os, joblib

class ModelTrainer:
    """Train model models."""
    def train(self, model, X_train, Y_train):
        """Train the requested data."""
        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        return model, train_score
    
    def save_model(self, model, filepath):
        """Save model."""
        joblib.dump(model, filepath)
    
    def load_model(self, filepath):
        """Load model."""
        return joblib.load(filepath)


