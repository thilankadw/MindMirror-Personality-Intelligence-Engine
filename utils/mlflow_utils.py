"""Utilities for MLflow utils."""
import os
import logging
import re
import socket
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse, unquote

from utils.config import get_mlflow_config, get_mlflow_models, get_mlflow_model_by_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow tracking utilities for experiment management and model versioning"""
    
    def __init__(self):
        """Initialize the m lflow tracker."""
        self.config = get_mlflow_config()
        self.setup_mlflow()

    def _tracking_uri_to_local_path(self, uri: str) -> Optional[Path]:
        """Resolve MLflow file-based tracking URI to a local filesystem path."""
        if not isinstance(uri, str):
            return None

        raw = uri.strip()
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            path_str = unquote(parsed.path or "")
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", path_str):
                path_str = path_str[1:]
            path = Path(path_str) if path_str else Path("./mlruns")
            return path.expanduser()

        if raw.startswith("file:"):
            path_str = raw[len("file:"):].strip()
            path = Path(path_str) if path_str else Path("./mlruns")
            return path.expanduser()

        return None

    def _ensure_directory_path(self, preferred_path: Path) -> Path:
        """
        Ensure a usable directory exists for MLflow file store.
        If the preferred path is an existing file, choose a safe sibling directory.
        """
        def _is_occupied_non_dir(path: Path) -> bool:
            # Use lexists so broken symlinks are treated as occupied paths too.
            """Return whether occupied non dir."""
            if not os.path.lexists(str(path)):
                return False
            return not path.is_dir()

        candidate = preferred_path
        if _is_occupied_non_dir(candidate):
            logger.warning(
                "MLflow local tracking path is not a directory: %s. Using sibling directory.",
                candidate,
            )
            candidate = preferred_path.with_name(f"{preferred_path.name}_dir")

        attempt = 0
        while attempt < 20:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                if candidate.is_dir():
                    return candidate
            except FileExistsError:
                # Path became occupied between checks; retry with a different name.
                if candidate.is_dir():
                    return candidate

            attempt += 1
            candidate = preferred_path.with_name(f"{preferred_path.name}_dir_{attempt}")

        raise OSError(
            f"Unable to create MLflow tracking directory from base path: {preferred_path}"
        )

    def _prepare_tracking_uri(self, uri: str) -> str:
        """
        Normalize tracking URI for reliable local fallback behavior.
        For file URIs: ensure backing directory exists and return an absolute file URI.
        """
        local_path = self._tracking_uri_to_local_path(uri)
        if local_path is None:
            return uri

        # In the MLflow container, prefer the dedicated mounted store path to avoid
        # edge cases on the project bind mount (/workspace).
        if not local_path.is_absolute():
            container_store = Path("/mlflow/mlruns")
            if container_store.is_dir():
                return container_store.resolve().as_uri()

        local_dir = self._ensure_directory_path(local_path)
        return local_dir.resolve().as_uri()

    def _is_http_tracking_uri(self, uri: str) -> bool:
        """Return whether HTTP tracking URI."""
        if not isinstance(uri, str):
            return False
        u = uri.strip().lower()
        return u.startswith("http://") or u.startswith("https://")

    def _is_tracking_endpoint_reachable(self, uri: str, timeout: float = 0.8) -> bool:
        """Fast TCP reachability check to avoid long MLflow retry delays."""
        try:
            parsed = urlparse(uri)
            host = parsed.hostname
            if not host:
                return False
            if parsed.port:
                port = parsed.port
            elif parsed.scheme == "https":
                port = 443
            else:
                port = 80

            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _normalize_uri_for_compare(self, uri: Optional[str]) -> str:
        """Normalize URI for compare."""
        if not uri:
            return ""
        return str(uri).strip().rstrip("/")

    def _prepare_experiment(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        artifact_root: Optional[str],
    ) -> Optional[mlflow.entities.Experiment]:
        """
        Ensure the target experiment exists and is selected.
        For HTTP tracking URIs, create the experiment with an explicit artifact
        root when configured so artifacts are routed to S3.
        """
        client = mlflow.tracking.MlflowClient()
        normalized_artifact_root = self._normalize_uri_for_compare(artifact_root)
        use_explicit_artifact_root = bool(
            normalized_artifact_root and self._is_http_tracking_uri(tracking_uri)
        )

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None and use_explicit_artifact_root:
            try:
                experiment_id = client.create_experiment(
                    name=experiment_name,
                    artifact_location=normalized_artifact_root,
                )
                experiment = client.get_experiment(experiment_id)
                logger.info(
                    "Created MLflow experiment '%s' with artifact root: %s",
                    experiment_name,
                    normalized_artifact_root,
                )
            except Exception as exc:
                logger.warning(
                    "Could not create experiment '%s' with explicit artifact root %s: %s",
                    experiment_name,
                    normalized_artifact_root,
                    exc,
                )

        mlflow.set_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment and use_explicit_artifact_root:
            current_location = self._normalize_uri_for_compare(experiment.artifact_location)
            if current_location and current_location != normalized_artifact_root:
                logger.warning(
                    "MLflow experiment '%s' artifact location is '%s' (expected '%s'). "
                    "Existing experiments keep their original artifact location.",
                    experiment_name,
                    current_location,
                    normalized_artifact_root,
                )
            else:
                logger.info(
                    "MLflow experiment '%s' artifact location: %s",
                    experiment_name,
                    current_location or normalized_artifact_root,
                )

        return experiment
        
    def setup_mlflow(self):
        """Initialize MLflow tracking with configuration"""
        primary_uri = self.config.get('tracking_uri', 'file:./mlruns')
        fallback_uri = self.config.get('fallback_tracking_uri')
        experiment_name = self.config.get('experiment_name', 'Big5_Trait_Prediction')
        artifact_root = self.config.get("artifact_root")

        candidate_uris = [primary_uri]
        if fallback_uri and fallback_uri not in candidate_uris:
            candidate_uris.append(fallback_uri)
        # Safety net: always include a local file-store fallback when configured
        # URIs are remote endpoints (e.g., localhost server not running).
        if not any(self._tracking_uri_to_local_path(uri) is not None for uri in candidate_uris):
            candidate_uris.append("file:./mlruns")

        last_error = None
        for uri in candidate_uris:
            try:
                if self._is_http_tracking_uri(uri) and not self._is_tracking_endpoint_reachable(uri):
                    logger.warning("Skipping unreachable MLflow tracking_uri=%s", uri)
                    continue
                prepared_uri = self._prepare_tracking_uri(uri)
                mlflow.set_tracking_uri(prepared_uri)
                experiment = self._prepare_experiment(
                    tracking_uri=prepared_uri,
                    experiment_name=experiment_name,
                    artifact_root=artifact_root,
                )
                experiment_id = experiment.experiment_id if experiment else "unknown"
                logger.info(
                    "Using MLflow experiment: %s (ID: %s) on %s",
                    experiment_name,
                    experiment_id,
                    prepared_uri,
                )
                return
            except Exception as e:
                last_error = e
                logger.warning("MLflow setup failed for tracking_uri=%s: %s", uri, e)

        logger.error("Error setting up MLflow experiment: %s", last_error)
        raise last_error
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        # Check if there's already an active run
        active_run = mlflow.active_run()
        if active_run is not None:
            logger.warning(f"Active run detected (ID: {active_run.info.run_id}). Ending it before starting new run.")
            mlflow.end_run()
        
        # Format timestamp for run name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if run_name is None:
            run_name_prefix = self.config.get('run_name_prefix', 'run')
            run_name_prefix = run_name_prefix.replace('_', ' ')
            base_run_name = run_name_prefix
        else:
            base_run_name = run_name.replace('_', ' ')

        # Build model-aware run name so multiple models are clearly separated in MLflow UI.
        name_parts = []
        if tags:
            model_key = tags.get('model_key')
            trait = tags.get('trait')
            pipeline_type = tags.get('pipeline_type')
            if model_key:
                name_parts.append(str(model_key))
            elif trait:
                name_parts.append(str(trait).replace('_', ' '))
            if pipeline_type:
                name_parts.append(str(pipeline_type).replace('_', ' '))
        name_parts.append(base_run_name)
        run_name = " | ".join(name_parts) + f" | {timestamp}"
        
        # Merge default tags with provided tags
        default_tags = self.config.get('tags', {})
        if tags:
            default_tags.update(tags)
            
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        print(f"MLflow Run Name: {run_name}")
        return run
    
    def log_data_pipeline_metrics(self, dataset_info: Dict[str, Any]):
        """Log data pipeline metrics and artifacts"""
        try:
            mlflow.log_metrics({
                'dataset_rows': dataset_info.get('total_rows', 0),
                'training_rows': dataset_info.get('train_rows', 0),
                'test_rows': dataset_info.get('test_rows', 0),
                'num_features': dataset_info.get('num_features', 0),
            })
            
            mlflow.log_params({
                'test_size': dataset_info.get('test_size', 0.2),
                'random_state': dataset_info.get('random_state', 42),
                'preprocessing_steps': dataset_info.get('preprocessing_steps', [])
            })

            if 'feature_names' in dataset_info:
                mlflow.log_param('feature_names', str(dataset_info['feature_names']))

            if dataset_info.get('tau') is not None:
                mlflow.log_metric('tau', dataset_info['tau'])
            if dataset_info.get('tau_quantile') is not None:
                mlflow.log_param('tau_quantile', dataset_info['tau_quantile'])

            if dataset_info.get('authors_train') is not None:
                mlflow.log_metric('authors_train', dataset_info['authors_train'])
            if dataset_info.get('authors_test') is not None:
                mlflow.log_metric('authors_test', dataset_info['authors_test'])

            logger.info("Logged data pipeline metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging data pipeline metrics: {e}")
    
    def log_training_metrics(self, model, training_metrics: Dict[str, Any], model_params: Dict[str, Any]):
        """Log training metrics, parameters, and model artifacts."""
        try:
            mlflow.log_params(model_params)
            
            if 'overall_metrics' in training_metrics:
                overall = training_metrics['overall_metrics']
                if 'accuracy' in overall:
                    mlflow.log_metrics({
                        'overall_accuracy': overall.get('accuracy', 0),
                        'overall_balanced_accuracy': overall.get('balanced_accuracy', 0),
                        'overall_f1_macro': overall.get('f1_macro', 0),
                        'overall_f1_weighted': overall.get('f1_weighted', 0),
                        'overall_precision_weighted': overall.get('precision_weighted', 0),
                        'overall_recall_weighted': overall.get('recall_weighted', 0),
                    })
                else:
                    regression_metrics = {
                        'overall_mae': overall.get('mae', 0),
                        'overall_mse': overall.get('mse', 0),
                        'overall_rmse': overall.get('rmse', 0),
                        'overall_r2': overall.get('r2', 0),
                    }
                    if overall.get('pearson_r') is not None:
                        regression_metrics['overall_pearson_r'] = overall.get('pearson_r')
                    mlflow.log_metrics(regression_metrics)
            
            if 'per_target_metrics' in training_metrics:
                for target_metrics in training_metrics['per_target_metrics']:
                    target = target_metrics['target'].replace('delta_', '')
                    per_target_log = {
                        f'{target}_mae': target_metrics.get('mae', 0),
                        f'{target}_rmse': target_metrics.get('rmse', 0),
                        f'{target}_r2': target_metrics.get('r2', 0),
                    }
                    if target_metrics.get('pearson_r') is not None:
                        per_target_log[f'{target}_pearson_r'] = target_metrics.get('pearson_r')
                    mlflow.log_metrics(per_target_log)
            
            mlflow.log_params({
                'n_samples': training_metrics.get('n_samples', 0),
                'n_targets': training_metrics.get('n_targets', 0)
            })
            
            eval_report_path = 'evaluation_report.json'
            with open(eval_report_path, 'w') as f:
                json.dump(training_metrics, f, indent=4)
            mlflow.log_artifact(eval_report_path, "evaluation")
            os.remove(eval_report_path)
            
            artifact_path = self.config.get('artifact_path', 'model')
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
            )
            
            logger.info("Logged training metrics and model to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")
    
    def log_evaluation_metrics(self, evaluation_metrics: Dict[str, Any]):
        """Log evaluation metrics and artifacts"""
        try:
            if 'overall_metrics' in evaluation_metrics:
                overall = evaluation_metrics['overall_metrics']
                if 'accuracy' in overall:
                    mlflow.log_metrics({
                        'eval_accuracy': overall.get('accuracy', 0),
                        'eval_balanced_accuracy': overall.get('balanced_accuracy', 0),
                        'eval_f1_macro': overall.get('f1_macro', 0),
                        'eval_f1_weighted': overall.get('f1_weighted', 0),
                        'eval_precision_weighted': overall.get('precision_weighted', 0),
                        'eval_recall_weighted': overall.get('recall_weighted', 0),
                    })
                else:
                    eval_regression_metrics = {
                        'eval_mae': overall.get('mae', 0),
                        'eval_rmse': overall.get('rmse', 0),
                        'eval_r2': overall.get('r2', 0),
                    }
                    if overall.get('pearson_r') is not None:
                        eval_regression_metrics['eval_pearson_r'] = overall.get('pearson_r')
                    mlflow.log_metrics(eval_regression_metrics)
            
            logger.info("Logged evaluation metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging evaluation metrics: {e}")
    
    def log_inference_metrics(self, predictions: np.ndarray, input_data_info: Optional[Dict[str, Any]] = None):
        """Log inference metrics and results for trait change predictions"""
        try:
            inference_metrics = {
                'num_predictions': len(predictions),
                'num_targets': predictions.shape[1] if predictions.ndim > 1 else 1
            }
            
            if predictions.ndim > 1:
                for i in range(predictions.shape[1]):
                    target_predictions = predictions[:, i]
                    inference_metrics.update({
                        f'target_{i}_mean': float(np.mean(target_predictions)),
                        f'target_{i}_std': float(np.std(target_predictions)),
                        f'target_{i}_min': float(np.min(target_predictions)),
                        f'target_{i}_max': float(np.max(target_predictions))
                    })
            
            mlflow.log_metrics(inference_metrics)
            
            if input_data_info:
                mlflow.log_params(input_data_info)
            
            logger.info("Logged inference metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging inference metrics: {e}")
    
    def load_model_from_registry(self, model_name: Optional[str] = None, 
                               version: Optional[Union[int, str]] = None, 
                               stage: Optional[str] = None):
        """Load model from MLflow Model Registry"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'big5_classification_model')
            
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow registry: {e}")
            return None
    
    def get_latest_model_version(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the latest version of a registered model"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'big5_classification_model')
            
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            
            if latest_version:
                return latest_version[0].version
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    def transition_model_stage(self, model_name: Optional[str] = None, 
                             version: Optional[str] = None, 
                             stage: str = "Staging"):
        """Transition model to a specific stage"""
        try:
            if model_name is None:
                model_name = self.config.get('model_registry_name', 'big5_classification_model')
            
            if version is None:
                version = self.get_latest_model_version(model_name)
            
            if version:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                logger.info(f"Transitioned model {model_name} version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

def setup_mlflow_autolog():
    """Setup MLflow autologging for supported frameworks"""
    mlflow_config = get_mlflow_config()
    autolog_enabled = mlflow_config.get('autolog')
    if autolog_enabled is None:
        autolog_enabled = mlflow_config.get('run_policy', {}).get('autolog', True)
    if autolog_enabled:
        mlflow.sklearn.autolog()
        logger.info("MLflow autologging enabled for scikit-learn")

def create_mlflow_run_tags(pipeline_type: str, additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create standardized tags for MLflow runs"""
    tags = {
        'pipeline_type': pipeline_type,
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_tags:
        tags.update(additional_tags)

    def _infer_trait_from_target_type(target_type: str) -> Optional[str]:
        """Handle infer trait from target type."""
        if not target_type:
            return None
        t = str(target_type).strip().lower()
        if t.endswith("_direction"):
            return t[: -len("_direction")]
        if t.endswith("_dir"):
            return t[: -len("_dir")]
        if t.endswith("_state"):
            return t[: -len("_state")]
        return None

    model_key = tags.get("model_key")
    registry_name = tags.get("registry_name")
    trait = tags.get("trait")
    domain = tags.get("domain")
    target_type = tags.get("target_type")

    if not trait:
        trait = _infer_trait_from_target_type(str(target_type)) if target_type else None
    if trait and not domain:
        domain = "personality"

    if model_key is None or registry_name is None:
        candidates = get_mlflow_models()
        resolved = None

        # 1) explicit model key from tag, if present
        if model_key:
            resolved = next((m for m in candidates if m.get("key") == model_key), None)

        # 2) personality trait lookup
        if resolved is None and domain == "personality" and trait:
            target_name = f"{trait}_dir"
            resolved = next(
                (
                    m
                    for m in candidates
                    if m.get("domain") == "personality"
                    and (m.get("target") == target_name or m.get("key") == f"big5_{trait}_state")
                ),
                None,
            )

        # 3) fallback by target tag if it already matches configured model target
        if resolved is None and target_type:
            resolved = next((m for m in candidates if m.get("target") == target_type), None)

        if resolved:
            model_key = model_key or resolved.get("key")
            registry_name = registry_name or resolved.get("registry_name")
            domain = domain or resolved.get("domain")

    if model_key:
        tags["model_key"] = model_key
    if registry_name:
        tags["registry_name"] = registry_name
    if trait:
        tags["trait"] = trait
    if domain:
        tags["domain"] = domain
    
    return tags


def register_active_run_model(
    *,
    model_key: str,
    default_registry_name: Optional[str] = None,
    default_artifact_path: str = "model",
    tags: Optional[Dict[str, str]] = None,
    alias: str = "challenger",
) -> Dict[str, str]:
    """
    Register the current active run's logged model artifact into MLflow Model Registry.
    Returns registration metadata including registry_name, version and model_uri.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Cannot register model: no active MLflow run found.")

    model_cfg = get_mlflow_model_by_key(model_key) or {}
    registry_name = str(model_cfg.get("registry_name") or default_registry_name or "").strip()
    if not registry_name:
        raise RuntimeError(f"Cannot register model: registry_name is missing for model_key='{model_key}'.")

    artifact_path = str(model_cfg.get("artifact_path") or default_artifact_path or "model").strip()
    if not artifact_path:
        artifact_path = "model"

    run_id = active_run.info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    client = mlflow.tracking.MlflowClient()

    try:
        client.create_registered_model(registry_name)
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" not in message and "resource_already_exists" not in message:
            raise

    registered = mlflow.register_model(model_uri=model_uri, name=registry_name)
    version = str(registered.version)

    model_tags = {
        "model_key": model_key,
        "run_id": run_id,
    }
    if tags:
        model_tags.update({k: str(v) for k, v in tags.items() if v is not None})
    for key, value in model_tags.items():
        client.set_model_version_tag(registry_name, version, key, str(value))

    if alias:
        try:
            client.set_registered_model_alias(registry_name, alias, registered.version)
        except Exception as exc:
            logger.warning(
                "Registered model version created, but failed to set alias '%s' on '%s': %s",
                alias,
                registry_name,
                exc,
            )

    logger.info(
        "Registered MLflow model version: model_key=%s name=%s version=%s source=%s",
        model_key,
        registry_name,
        version,
        model_uri,
    )
    return {
        "model_key": model_key,
        "registry_name": registry_name,
        "version": version,
        "model_uri": model_uri,
        "artifact_path": artifact_path,
        "run_id": run_id,
    }
