"""Training pipeline for conscientiousness."""
import os
import json
from typing import Any, Dict, Optional
import logging
import pandas as pd
import mlflow
import inspect
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pipelines.personality.conscientiousness.data_pipeline import data_pipeline
from utils.config import get_data_paths, get_model_config, get_mean_features, get_std_features, get_target_features, get_environment_config, get_mlflow_config, get_temporal_features, get_task_type
from src.personality.model_building import HistGradientBoostingModelBuilder, RandomForestModelBuilder, XGBoostModelBuilder
from src.personality.model_training import ModelTrainer
from src.personality.model_evaluation import ModelEvaluator
from utils.mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags, register_active_run_model

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DOMAIN = 'personality'
TRAIT = 'conscientiousness'

def _filter_model_params(model_cls, params: Dict[str, Any]) -> Dict[str, Any]:
	"""Filter model params."""
	allowed = set(inspect.signature(model_cls).parameters.keys())
	filtered = {k: v for k, v in params.items() if k in allowed}
	dropped = sorted(set(params.keys()) - set(filtered.keys()))
	if dropped:
		logger.warning("Dropping unsupported parameters for %s: %s", model_cls.__name__, dropped)
	return filtered

def training_pipeline(
		model_params: Optional[Dict[str, Any]] = None,
		model_path: Optional[str] = None
	):

	"""Handle training pipeline."""
	data_paths = get_data_paths(domain=DOMAIN, trait=TRAIT)
	x_train_path = data_paths.get('X_train', 'artifacts/data/personality/{TRAIT}/X_train.csv')
	x_test_path = data_paths.get('X_test', 'artifacts/data/personality/{TRAIT}/X_test.csv')
	y_train_path = data_paths.get('Y_train', 'artifacts/data/personality/{TRAIT}/Y_train.csv')
	y_test_path = data_paths.get('Y_test', 'artifacts/data/personality/{TRAIT}/Y_test.csv')

	def _splits_ready(*paths: str) -> bool:
		"""Handle splits ready."""
		for path in paths:
			if not path or not os.path.exists(path):
				return False
			if os.path.getsize(path) == 0:
				return False
		return True

	# Check if data exists first, before starting MLflow run
	if not _splits_ready(x_train_path, x_test_path, y_train_path, y_test_path):
		logger.info("Artifact files not found. Running data pipeline...")
		data_pipeline()
		if not _splits_ready(x_train_path, x_test_path, y_train_path, y_test_path):
			raise FileNotFoundError("Train/test split files are still missing after running data pipeline.")

	# Now start the training MLflow run
	mlflow_tracker = MLflowTracker()
	# Disable autologging to have full control over logged metrics
	# setup_mlflow_autolog()
	
	# Get configuration for tags
	env_config = get_environment_config(domain=DOMAIN, trait=TRAIT)
	mlflow_config = get_mlflow_config()
	model_config = get_model_config(domain=DOMAIN, trait=TRAIT)
	target_features = get_target_features(domain=DOMAIN, trait=TRAIT)
	
	model_type = model_config.get('model_type', 'xgboost')
	target_type = (
		target_features[0].replace('_dir', '_direction')
		if target_features and target_features[0].endswith('_dir')
		else (target_features[0] if target_features else 'target')
	)
	run_tags = create_mlflow_run_tags(
		'train_pipeline', {
			'model_type': model_type,
			'model_architecture': (
				'XGBClassifier'
				if model_type == 'xgboost'
				else ('RandomForestClassifier' if model_type == 'random_forest' else 'HistGradientBoostingClassifier')
			),
			'week_window': str(env_config.get('week_window', 1)),
			'dataset_version': env_config.get('dataset_version', 'v1.0'),
			'prediction_horizon': '1_week_ahead',
			'target_type': target_type,
			'num_targets': '1',
			'target_traits': ','.join(target_features)
		}
	)
	run_name = f"{mlflow_config.get('run_name_prefix', 'run')}_train_{env_config.get('week_window', 1)}w"
	run = mlflow_tracker.start_run(run_name=run_name, tags=run_tags)
	
	# Get feature and target column names
	mean_features = get_mean_features(domain=DOMAIN, trait=TRAIT)
	std_features = get_std_features(domain=DOMAIN, trait=TRAIT)
	temporal_features = get_temporal_features(domain=DOMAIN, trait=TRAIT)
	feature_cols = mean_features + std_features + temporal_features
	task_type = get_task_type(domain=DOMAIN, trait=TRAIT)
	
	# Load the data splits
	logger.info("Loading data splits...")
	try:
		X_train_df = pd.read_csv(x_train_path)
		X_test_df = pd.read_csv(x_test_path)
		Y_train_df = pd.read_csv(y_train_path)
		Y_test_df = pd.read_csv(y_test_path)
		X_train = X_train_df.values
		X_test = X_test_df.values
		Y_train = Y_train_df.values
		Y_test = Y_test_df.values
		if Y_train.ndim > 1 and Y_train.shape[1] == 1:
			Y_train = Y_train.ravel()
		if Y_test.ndim > 1 and Y_test.shape[1] == 1:
			Y_test = Y_test.ravel()
		label_encoder = None
		if model_type == 'xgboost':
			label_encoder = LabelEncoder()
			y_train_labels = np.asarray(Y_train)
			y_test_labels = np.asarray(Y_test)
			label_encoder.fit(y_train_labels)
			unknown = set(pd.unique(y_test_labels)) - set(label_encoder.classes_)
			if unknown:
				raise ValueError(f"Unknown labels in test set: {sorted(unknown)}")
			Y_train = label_encoder.transform(y_train_labels)
			Y_test = label_encoder.transform(y_test_labels)
		logger.info("Artifacts loaded: X_train=%s, X_test=%s, Y_train=%s, Y_test=%s",
					X_train.shape,
					X_test.shape,
					Y_train.shape,
					Y_test.shape)
		
		# Log datasets to MLflow
		mlflow.log_input(
			mlflow.data.from_pandas(
				X_train_df,
				source=x_train_path,
				name="training_features"
			),
			context="training"
		)
		mlflow.log_input(
			mlflow.data.from_pandas(
				Y_train_df,
				source=y_train_path,
				name="training_targets"
			),
			context="training"
		)
		mlflow.log_input(
			mlflow.data.from_pandas(
				X_test_df,
				source=x_test_path,
				name="test_features"
			),
			context="evaluation"
		)
		mlflow.log_input(
			mlflow.data.from_pandas(
				Y_test_df,
				source=y_test_path,
				name="test_targets"
			),
			context="evaluation"
		)
		
	except Exception as e:
		logger.exception("Failed to load artifact files: %s", e)
		raise
	
	# Build model
	if model_path is None:
		model_path = model_config.get('model_path')
	eval_results_path = model_config.get(
		'evaluation_path',
		(model_path or 'artifacts/models/model.joblib').replace('.joblib', '_evaluation.json')
	)
	if model_type == 'random_forest':
		logger.info("Building RandomForest classifier...")
		if model_params is None:
			model_params = model_config.get('random_forest_params', {})
		model_params = _filter_model_params(RandomForestClassifier, model_params)
		model_builder = RandomForestModelBuilder(**model_params)
	elif model_type == 'xgboost':
		logger.info("Building XGBoost classifier...")
		if model_params is None:
			model_params = model_config.get('xgboost_params', {})
		model_builder = XGBoostModelBuilder(**model_params)
	else:
		logger.info("Building HistGradientBoosting classifier...")
		if model_params is None:
			model_params = model_config.get('hist_gradient_boosting_params', {})
		model_builder = HistGradientBoostingModelBuilder(**model_params)
	model = model_builder.build_model()
	logger.info("Model built successfully: %s", model_builder.model_name)
	
	# Train model
	logger.info("Training model...")
	trainer = ModelTrainer()
	model, train_score = trainer.train(model, X_train, Y_train)
	logger.info("Model training completed. Train score (accuracy): %.4f", train_score)
	
	# Evaluate model
	logger.info("Evaluating model...")
	evaluator = ModelEvaluator(
		model,
		model_builder.model_name,
		target_names=target_features,
		label_encoder=label_encoder,
	)
	eval_results = evaluator.evaluate(X_test, Y_test)
	
	# Save evaluation results
	evaluator.save_evaluation_results(eval_results_path)
	
	# Save model
	logger.info("Saving model to %s...", model_path)
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	model_builder.save_model(model_path)
	logger.info("Model saved successfully")
	
	# Log final summary
	logger.info("="*80)
	logger.info("Training Pipeline Completed Successfully!")
	logger.info("="*80)
	logger.info("Model: %s", model_builder.model_name)
	logger.info("Model Path: %s", model_path)
	logger.info("Evaluation Results Path: %s", eval_results_path)
	logger.info("Train Score (accuracy): %.4f", train_score)
	logger.info("Test Accuracy: %.6f", eval_results['overall_metrics']['accuracy'])
	logger.info("Test Balanced Accuracy: %.6f", eval_results['overall_metrics']['balanced_accuracy'])
	logger.info("Test F1 Macro: %.6f", eval_results['overall_metrics']['f1_macro'])
	logger.info("Test F1 Weighted: %.6f", eval_results['overall_metrics']['f1_weighted'])
	logger.info("="*80)
	
	# Log MLflow metrics and end run
	mlflow_tracker.log_training_metrics(model, eval_results, model_params)
	mlflow.log_params({
		'task_type': task_type,
		'target_features': ','.join(target_features),
		'feature_names': ','.join(X_train_df.columns.tolist()),
		'week_window': env_config.get('week_window', 1),
		'dataset_version': env_config.get('dataset_version', 'v1.0'),
		'num_features': X_train.shape[1],
		'num_samples_train': X_train.shape[0],
		'num_samples_test': X_test.shape[0]
	})

	if model_type == 'xgboost' and label_encoder is not None:
		label_encoder_path = model_config.get(
			'label_encoder_path',
			(model_path or 'artifacts/models/xgboost_classifier.joblib').replace('.joblib', '_label_encoder.joblib')
		)
		os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
		joblib.dump(label_encoder, label_encoder_path)
		mlflow.log_artifact(label_encoder_path, "model")
		mlflow.log_param('label_encoder_classes', ','.join(label_encoder.classes_.astype(str)))

	metadata_path = data_paths.get('data_pipeline_metadata')
	if metadata_path and os.path.exists(metadata_path):
		try:
			with open(metadata_path, 'r', encoding='utf-8') as f:
				metadata = json.load(f)
			if metadata.get('tau') is not None:
				mlflow.log_metric('tau', metadata['tau'])
			if metadata.get('tau_quantile') is not None:
				mlflow.log_param('tau_quantile', metadata['tau_quantile'])
			if metadata.get('authors_train') is not None:
				mlflow.log_metric('authors_train', metadata['authors_train'])
			if metadata.get('authors_test') is not None:
				mlflow.log_metric('authors_test', metadata['authors_test'])
		except Exception as e:
			logger.warning("Failed to read data pipeline metadata: %s", e)

	registration_info = register_active_run_model(
		model_key=f"big5_{TRAIT}_state",
		default_registry_name=f"mm_{TRAIT}_state",
		tags={
			"domain": DOMAIN,
			"trait": TRAIT,
			"pipeline_type": "train_pipeline",
		},
	)
	mlflow_tracker.end_run()
	
	# Return results summary
	return {
		'model_path': model_path,
		'evaluation_results_path': eval_results_path,
		'evaluation_metrics': eval_results,
		'train_score': train_score,
		'mlflow_registration': registration_info
	}

def main() -> None:
	"""Run the module entry point."""
	model_config = get_model_config(domain=DOMAIN, trait=TRAIT)
	model_type = model_config.get('model_type', 'xgboost')
	params_key = (
		'xgboost_params'
		if model_type == 'xgboost'
		else ('random_forest_params' if model_type == 'random_forest' else 'hist_gradient_boosting_params')
	)
	training_pipeline(model_params=model_config.get(params_key))

if __name__ == '__main__':
	main()



