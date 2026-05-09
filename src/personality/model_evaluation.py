"""Utilities for model evaluation."""
import numpy as np
import pandas as pd
import logging
import json
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model models."""
    def __init__(self, model, model_name, target_names=None, label_encoder=None):
        """Initialize the model evaluator."""
        self.model = model
        self.model_name = model_name
        self.target_names = target_names
        self.label_encoder = label_encoder
        self.evaluation_results = {}

    def evaluate(self, X_test, Y_test):
        """Evaluate the requested data."""
        logger.info("Starting model evaluation...")

        y_true = np.asarray(Y_test).ravel()
        y_pred = self.model.predict(X_test)

        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            if y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
            else:
                y_pred = np.argmax(y_pred, axis=1)

        if self.label_encoder is not None:
            try:
                y_true = self.label_encoder.inverse_transform(np.asarray(y_true, dtype=int))
                y_pred = self.label_encoder.inverse_transform(np.asarray(y_pred, dtype=int))
            except Exception as e:
                logger.warning("Label decoding failed; using raw labels. %s", e)

        labels = sorted(pd.unique(y_true))
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        self.evaluation_results = {
            "model_name": self.model_name,
            "n_samples": int(len(y_true)),
            "n_targets": 1,
            "overall_metrics": {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
            },
            "labels": [str(l) for l in labels],
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        logger.info("=" * 80)
        logger.info("Model Evaluation Results for: %s", self.model_name)
        logger.info("=" * 80)
        logger.info("Test samples: %d | Number of targets: %d", len(y_true), 1)
        logger.info("")
        logger.info("Overall Metrics:")
        logger.info("  Accuracy:           %.6f", accuracy)
        logger.info("  Balanced Accuracy:  %.6f", balanced_accuracy)
        logger.info("  F1 Macro:           %.6f", f1_macro)
        logger.info("  F1 Weighted:        %.6f", f1_weighted)
        logger.info("  Precision Weighted: %.6f", precision_weighted)
        logger.info("  Recall Weighted:    %.6f", recall_weighted)
        logger.info("  Confusion Matrix:   %s", cm.tolist())
        logger.info("=" * 80)

        return self.evaluation_results

    def save_evaluation_results(self, filepath: str):
        """Save evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save. Run evaluate() first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)

        logger.info("Evaluation results saved to: %s", filepath)
