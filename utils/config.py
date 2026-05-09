"""Configuration helpers."""
import os
import logging
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

PERSONALITY_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def load_config() -> Dict[str, Any]:
    """Load config."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        return {}


def _merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the requested data."""
    merged = dict(base or {})
    if override:
        merged.update(override)
    return merged


def _normalize_mlflow_artifact_root(raw_value: Optional[str]) -> Optional[str]:
    """
    Normalize MLflow artifact roots to URI form accepted by MLflow.
    Supports S3 ARN input by converting to s3://bucket[/prefix].
    """
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None

    arn_prefix = "arn:aws:s3:::"
    if value.startswith(arn_prefix):
        suffix = value[len(arn_prefix):].lstrip("/")
        return f"s3://{suffix}" if suffix else None

    return value


def _get_personality_trait_config(trait: Optional[str]) -> Dict[str, Any]:
    """Get personality trait config."""
    if not trait:
        return {}
    personality_cfg = get_domain_config("personality")
    return personality_cfg.get("traits", {}).get(trait, {})


# ==================== Core Getters ====================
def get_project_config() -> Dict[str, Any]:
    """Get project config."""
    return load_config().get("project", {})


def get_services_config() -> Dict[str, Any]:
    """Get services config."""
    return load_config().get("services", {})


def get_db_config() -> Dict[str, Any]:
    """Get database config."""
    return load_config().get("db", {})


def get_team_config() -> Dict[str, Any]:
    """Get team config."""
    return load_config().get("team", {})


def get_artifacts_config() -> Dict[str, Any]:
    """Get artifacts config."""
    return load_config().get("artifacts", {})


def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline config."""
    return load_config().get("pipeline", {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging config."""
    return load_config().get("logging", {})


def get_mlflow_config() -> Dict[str, Any]:
    """Get MLflow config."""
    cfg = load_config().get("mlflow", {})
    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    env_fallback_tracking_uri = os.getenv("MLFLOW_FALLBACK_TRACKING_URI")
    env_artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT") or os.getenv("AWS_S3")
    if env_tracking_uri:
        cfg["tracking_uri"] = env_tracking_uri
    if env_fallback_tracking_uri:
        cfg["fallback_tracking_uri"] = env_fallback_tracking_uri
    normalized_artifact_root = _normalize_mlflow_artifact_root(
        env_artifact_root or cfg.get("artifact_root")
    )
    if normalized_artifact_root:
        cfg["artifact_root"] = normalized_artifact_root
    run_policy = cfg.get("run_policy", {})
    if "run_name_prefix" not in cfg and run_policy.get("parent_run_name_prefix"):
        cfg["run_name_prefix"] = run_policy["parent_run_name_prefix"]
    return cfg


def get_inference_loading_config() -> Dict[str, Any]:
    """Get inference loading config."""
    return load_config().get("inference_loading", {})


def get_environment_config(domain: str = "personality", trait: Optional[str] = None) -> Dict[str, Any]:
    """Get environment config."""
    cfg = load_config()
    env_cfg = dict(cfg.get("environment", {}))

    if "week_window" not in env_cfg:
        env_cfg["week_window"] = 1
    if "dataset_version" not in env_cfg:
        env_cfg["dataset_version"] = "v1.0"

    project_cfg = get_project_config()
    if "timezone" not in env_cfg and project_cfg.get("timezone"):
        env_cfg["timezone"] = project_cfg["timezone"]

    domain_cfg = get_domain_config(domain)
    env_cfg = _merge(env_cfg, domain_cfg.get("environment", {}))
    if domain == "personality" and trait:
        env_cfg = _merge(env_cfg, _get_personality_trait_config(trait).get("environment", {}))
    return env_cfg


def get_direction_config(domain: str = "personality", trait: Optional[str] = None) -> Dict[str, Any]:
    """Get direction config."""
    if domain != "personality" or not trait:
        return {}
    return _get_personality_trait_config(trait).get("direction", {})


def get_agreeableness_direction_config() -> Dict[str, Any]:
    """Get agreeableness direction config."""
    return get_direction_config(domain="personality", trait="agreeableness")


def get_openness_direction_config() -> Dict[str, Any]:
    """Get openness direction config."""
    return get_direction_config(domain="personality", trait="openness")


def get_conscientiousness_direction_config() -> Dict[str, Any]:
    """Get conscientiousness direction config."""
    return get_direction_config(domain="personality", trait="conscientiousness")


def get_extraversion_direction_config() -> Dict[str, Any]:
    """Get extraversion direction config."""
    return get_direction_config(domain="personality", trait="extraversion")


def get_neuroticism_direction_config() -> Dict[str, Any]:
    """Get neuroticism direction config."""
    return get_direction_config(domain="personality", trait="neuroticism")


# ==================== Domain Getters ====================
def get_domain_config(domain: str = "personality") -> Dict[str, Any]:
    """Get domain config."""
    return load_config().get("domains", {}).get(domain, {})


def get_domain_artifacts(domain: str = "personality") -> Dict[str, Any]:
    """Get domain artifacts."""
    artifacts = get_artifacts_config()
    return artifacts.get("domains", {}).get(domain, {})


def get_domain_data_paths(domain: str = "personality") -> Dict[str, Any]:
    """Get domain data paths."""
    return get_domain_config(domain).get("data_paths", {})


def get_domain_preprocessing(domain: str = "personality") -> Dict[str, Any]:
    """Get domain preprocessing."""
    return get_domain_config(domain).get("preprocessing", {})


def get_domain_features(domain: str = "personality") -> Dict[str, Any]:
    """Get domain features."""
    return get_domain_config(domain).get("features", {})


def get_domain_targets(domain: str = "personality") -> Dict[str, Any]:
    """Get domain targets."""
    return get_domain_config(domain).get("targets", {})


def get_domain_model_variant_config(
    domain: str = "personality",
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Get domain model variant config."""
    if not variant:
        return {}
    return get_domain_config(domain).get("models", {}).get(variant, {})


def get_resolved_domain_features(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Get resolved domain features."""
    features = dict(get_domain_features(domain))
    if variant:
        features = _merge(
            features,
            get_domain_model_variant_config(domain=domain, variant=variant).get("features", {}),
        )
    if domain == "personality" and trait:
        features = _merge(features, _get_personality_trait_config(trait).get("features", {}))
    return features


def get_configured_feature_columns(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[str]:
    """Get configured feature columns."""
    features = get_resolved_domain_features(domain=domain, trait=trait, variant=variant)
    configured = features.get("inference_columns") or features.get("feature_columns")
    if isinstance(configured, list):
        return list(configured)
    return []


# ==================== MLflow Models Getters ====================
def get_mlflow_models() -> List[Dict[str, Any]]:
    """Get MLflow models."""
    return get_mlflow_config().get("models", [])


def get_mlflow_models_for_domain(domain: str) -> List[Dict[str, Any]]:
    """Get MLflow models for domain."""
    return [m for m in get_mlflow_models() if m.get("domain") == domain]


def get_mlflow_model_by_key(key: str) -> Optional[Dict[str, Any]]:
    """Get MLflow model by key."""
    for model in get_mlflow_models():
        if model.get("key") == key:
            return model
    return None


# ==================== Legacy Domain-Specific Getters ====================
def get_data_paths(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Get data paths."""
    data_paths = dict(get_domain_data_paths(domain))
    if variant:
        data_paths = _merge(
            data_paths,
            get_domain_model_variant_config(domain=domain, variant=variant).get("data_paths", {}),
        )
    if domain == "personality" and trait:
        trait_cfg = _get_personality_trait_config(trait)
        data_paths = _merge(data_paths, trait_cfg.get("data_paths", {}))

    # Common compatibility keys used by existing pipelines
    raw_dir = data_paths.get("raw_dir", "data/raw/personality")
    processed_dir = data_paths.get("processed_dir", "data/processed/personality")
    if "data_path" not in data_paths:
        data_paths["data_path"] = data_paths.get("weekly_dataset", "")
    if "raw_json" not in data_paths:
        data_paths["raw_json"] = os.path.join(raw_dir, "reddit_post_data.json")
    if trait:
        data_paths.setdefault("data_artifacts_dir", f"artifacts/data/personality/{trait}")
        data_paths.setdefault("processed_data_dir", processed_dir)
        data_paths.setdefault("X_train", f"artifacts/data/personality/{trait}/X_train.csv")
        data_paths.setdefault("X_test", f"artifacts/data/personality/{trait}/X_test.csv")
        data_paths.setdefault("Y_train", f"artifacts/data/personality/{trait}/Y_train.csv")
        data_paths.setdefault("Y_test", f"artifacts/data/personality/{trait}/Y_test.csv")
        data_paths.setdefault(
            "data_pipeline_metadata",
            f"artifacts/data/personality/{trait}/data_pipeline_metadata.json",
        )

    return data_paths


def get_traits(domain: str = "personality") -> List[str]:
    """Get traits."""
    if domain != "personality":
        return []
    targets = get_domain_targets(domain)
    if targets:
        return list(targets.keys())
    return PERSONALITY_TRAITS


def get_analysis_traits(domain: str = "personality") -> List[str]:
    """Get analysis traits."""
    if domain == "personality":
        return get_traits(domain)
    return []


def get_required_attributes(domain: str = "personality") -> List[str]:
    """Get required attributes."""
    preprocessing = get_domain_preprocessing(domain)
    required = preprocessing.get("required_attributes")
    if isinstance(required, list) and required:
        return required
    return [
        "processed_text",
        "author",
        "created_date",
        "scores",
        "text_embeddings",
    ]


def get_mean_features(domain: str = "personality", trait: Optional[str] = None) -> List[str]:
    """Get mean features."""
    if domain != "personality":
        return []
    if trait:
        return [f"{trait}_mean_w"]
    return [f"{t}_mean_w" for t in get_traits(domain)]


def get_std_features(domain: str = "personality", trait: Optional[str] = None) -> List[str]:
    """Get std features."""
    if domain != "personality":
        return []
    if trait:
        return [f"{trait}_std_w"]
    return [f"{t}_std_w" for t in get_traits(domain)]


def get_text_embedding_feature(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[str]:
    """Get text embedding feature."""
    features = get_resolved_domain_features(domain=domain, trait=trait, variant=variant)
    embedding = features.get("embedding_feature_name", "text_embeddings_mean_w")
    return [embedding] if embedding else []


def get_temporal_features(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[str]:
    """Get temporal features."""
    features = get_resolved_domain_features(domain=domain, trait=trait, variant=variant)
    return features.get("temporal_features", [])


def get_grouping_feature(domain: str = "personality") -> List[str]:
    """Get grouping feature."""
    features = get_domain_features(domain)
    group_feature = features.get("group_feature", "author")
    return [group_feature] if group_feature else []


def get_time_sort_column(domain: str = "personality") -> str:
    """Get time sort column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("time_sort_column", "week_date")


def get_week_column(domain: str = "personality") -> str:
    """Get week column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("week_column", "week")


def get_created_date_column(domain: str = "personality") -> str:
    """Get created date column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("created_date_column", "created_date")


def get_processed_text_column(domain: str = "personality") -> str:
    """Get processed text column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("text_column", "processed_text")


def get_raw_embedding_column(domain: str = "personality") -> str:
    """Get raw embedding column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("raw_embedding_column", "text_embeddings")


def get_scores_column(domain: str = "personality") -> str:
    """Get scores column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("scores_column", "scores")


def get_title_column(domain: str = "personality") -> str:
    """Get title column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("title_column", "title")


def get_selftext_column(domain: str = "personality") -> str:
    """Get selftext column."""
    preprocessing = get_domain_preprocessing(domain)
    return preprocessing.get("selftext_column", "selftext")


def get_target_features(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[str]:
    """Get target features."""
    targets = get_domain_targets(domain)
    if domain == "personality" and trait:
        target = targets.get(trait)
        if target:
            return [target]
        trait_target = _get_personality_trait_config(trait).get("target")
        return [trait_target] if trait_target else []
    if variant:
        target = targets.get(variant)
        if isinstance(target, list):
            return list(target)
        if target:
            return [target]
        variant_target = get_domain_model_variant_config(domain=domain, variant=variant).get("target")
        if isinstance(variant_target, list):
            return list(variant_target)
        return [variant_target] if variant_target else []
    return list(targets.values()) if targets else []


# ==================== Training & Model Getters ====================
def get_training_config(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Get training config."""
    config = load_config()
    training_cfg = dict(config.get("training", {}))
    training_cfg = _merge(training_cfg, get_domain_config(domain).get("training", {}))
    if variant:
        training_cfg = _merge(
            training_cfg,
            get_domain_model_variant_config(domain=domain, variant=variant).get("training", {}),
        )
    if domain == "personality" and trait:
        training_cfg = _merge(training_cfg, _get_personality_trait_config(trait).get("training", {}))
    return training_cfg


def get_task_type(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """Get task type."""
    return get_training_config(domain=domain, trait=trait, variant=variant).get(
        "task_type",
        "classification",
    )


def get_model_config(
    domain: str = "personality",
    trait: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Get model config."""
    config = load_config()
    model_cfg = dict(config.get("model", {}))
    model_cfg = _merge(model_cfg, get_domain_config(domain).get("model", {}))
    if variant:
        model_cfg = _merge(
            model_cfg,
            get_domain_model_variant_config(domain=domain, variant=variant).get("model", {}),
        )
    if domain == "personality" and trait:
        model_cfg = _merge(model_cfg, _get_personality_trait_config(trait).get("model", {}))

        # MLflow fallback path for local inference/training artifacts
        if not model_cfg.get("model_path"):
            mlflow_model = get_mlflow_model_by_key(f"big5_{trait}_state")
            if mlflow_model and mlflow_model.get("local_fallback_path"):
                model_cfg["model_path"] = mlflow_model["local_fallback_path"]
        if not model_cfg.get("evaluation_path"):
            model_cfg["evaluation_path"] = f"artifacts/evaluation/personality/{trait}/evaluation.json"
        if not model_cfg.get("label_encoder_path"):
            model_cfg["label_encoder_path"] = (
                f"artifacts/models/personality/{trait}/label_encoder.joblib"
            )
    elif variant and not model_cfg.get("model_path"):
        mlflow_model = get_mlflow_model_by_key(f"{domain}_{variant}_model")
        if mlflow_model and mlflow_model.get("local_fallback_path"):
            model_cfg["model_path"] = mlflow_model["local_fallback_path"]
    return model_cfg


# ==================== Model Parameters Getters ====================
def get_random_forest_params() -> Dict[str, Any]:
    """Get random forest params."""
    return get_model_config().get("random_forest_params", {})


def get_hist_gradient_boosting_params() -> Dict[str, Any]:
    """Get hist gradient boosting params."""
    return get_model_config().get("hist_gradient_boosting_params", {})


def get_xgboost_params() -> Dict[str, Any]:
    """Get xgboost params."""
    return get_model_config().get("xgboost_params", {})


# ==================== Legacy Model Path Getters ====================
def get_model_path(domain: str = "personality", model_type: str = "xgboost") -> str:
    """Get model path."""
    domain_artifacts = get_domain_artifacts(domain)
    return domain_artifacts.get("models_dir", "artifacts/models")


def get_evaluation_path(domain: str = "personality") -> str:
    """Get evaluation path."""
    domain_artifacts = get_domain_artifacts(domain)
    return domain_artifacts.get("evaluation_dir", f"artifacts/evaluation/{domain}")


# ==================== Support Getters ====================
def get_bigfive_scoring_config() -> Dict[str, Any]:
    """Get bigfive scoring config."""
    return load_config().get("bigfive_scoring", {})


def get_text_embedding_config() -> Dict[str, Any]:
    """Get text embedding config."""
    return load_config().get("text_embedding", {})


def get_data_splitting() -> Dict[str, Any]:
    """Get data splitting."""
    return load_config().get("data_splitting", {})


def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation config."""
    return load_config().get("evaluation", {})


def get_deployment_config() -> Dict[str, Any]:
    """Get deployment config."""
    return load_config().get("deployment", {})


def get_inference_config() -> Dict[str, Any]:
    """Get inference config."""
    return load_config().get("inference", {})
