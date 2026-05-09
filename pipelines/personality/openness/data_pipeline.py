"""Data pipeline for openness."""
import os
import json
import logging
import pandas as pd
import numpy as np
import mlflow

from src.personality.data_parser import TraitScoreParser, TextEmbeddingParser
from src.personality.weekly_preprocess import TraitWeeklyAggregator, EmbeddingWeeklyAggregator, GapWeekCalculator
from src.personality.delta_targets import FutureDeltaTargetCreator, DirectionLabeler
from src.personality.lagged_features import (
    RollingStatsFeatureCreator,
    LagFeatureCreator,
    FeatureShiftCreator,
)
from utils.config import (
    get_data_paths,
    get_required_attributes,
    get_traits,
    get_openness_direction_config,
    get_mean_features,
    get_std_features,
    get_text_embedding_feature,
    get_grouping_feature,
    get_analysis_traits,
    get_week_column,
    get_time_sort_column,
    get_created_date_column,
    get_processed_text_column,
    get_raw_embedding_column,
    get_scores_column,
    get_title_column,
    get_selftext_column,
    get_target_features,
    get_environment_config,
    get_mlflow_config,
)
from src.personality.data_ingestion import DataIngestorCSV, DataIngestorJson
from src.personality.data_preprocess import AttributeConcatenation, DropAttributes, CleanText, ReplaceEmoji
from src.personality.vector_embedding import VectorEmbedding
from src.personality.data_splitter import AuthorTimeSplitStrategy
from utils.mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DOMAIN = 'personality'
TRAIT = 'openness'

def _log_df_state(df: pd.DataFrame, step: str):
    """Handle log data frame state."""
    if df is None:
        logger.info("%s: df is None", step)
        return
    try:
        null_counts = df.isna().sum()
        total_nulls = int(null_counts.sum())
        logger.info(
            "%s: shape=%s cols=%s total_nulls=%s",
            step,
            df.shape,
            df.columns.tolist(),
            total_nulls,
        )
        if total_nulls > 0:
            top_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
            logger.info("%s: nulls_by_col=%s", step, top_nulls.to_dict())
    except Exception as e:
        logger.warning("%s: failed to log df state: %s", step, e)

def data_pipeline(
    data_path: str = None,
    test_size: float = None,
    k_past: int = None,
    h_future: int = None,
    tau_quantile: float = None,
):
    """Handle data pipeline."""
    data_paths = get_data_paths(domain=DOMAIN, trait=TRAIT)
    required_attributes = get_required_attributes(domain=DOMAIN)
    traits = get_traits(domain=DOMAIN)
    mean_features = get_mean_features(domain=DOMAIN, trait=TRAIT)
    std_features = get_std_features(domain=DOMAIN, trait=TRAIT)
    text_embedding_features = get_text_embedding_feature(domain=DOMAIN)
    grouping_features = get_grouping_feature(domain=DOMAIN)
    analysis_traits = get_analysis_traits(domain=DOMAIN)
    target_features = get_target_features(domain=DOMAIN, trait=TRAIT)
    week_col = get_week_column(domain=DOMAIN)
    time_sort_col = get_time_sort_column(domain=DOMAIN)
    created_date_col = get_created_date_column(domain=DOMAIN)
    processed_text_col = get_processed_text_column(domain=DOMAIN)
    raw_embedding_col = get_raw_embedding_column(domain=DOMAIN)
    scores_col = get_scores_column(domain=DOMAIN)
    title_col = get_title_column(domain=DOMAIN)
    selftext_col = get_selftext_column(domain=DOMAIN)
    direction_cfg = get_openness_direction_config()
    if len(target_features) != 1:
        raise ValueError(
            "Expected exactly one target feature in config.yaml: attributes.target_features"
        )
    label_col = target_features[0]

    if len(mean_features) != 1:
        raise ValueError(
            "Expected exactly one mean feature in config.yaml: attributes.mean_features"
        )
    obs_col = mean_features[0]

    if len(text_embedding_features) != 1:
        raise ValueError(
            "Expected exactly one text embedding feature in config.yaml: "
            "attributes.text_embedding_feature"
        )
    emb_col = text_embedding_features[0]

    if len(grouping_features) != 1:
        raise ValueError(
            "Expected exactly one grouping feature in config.yaml: attributes.grouping_feature"
        )
    author_col = grouping_features[0]

    std_col = None
    if len(std_features) > 1:
        raise ValueError(
            "Expected at most one std feature in config.yaml: attributes.std_features"
        )
    if std_features:
        std_col = std_features[0]

    if test_size is None:
        test_size = direction_cfg.get("test_size", 0.2)
    if k_past is None:
        k_past = direction_cfg.get("k_past", 4)
    if h_future is None:
        h_future = direction_cfg.get("h_future", 4)
    if tau_quantile is None:
        tau_quantile = direction_cfg.get("tau_quantile", 0.7)

    # Resolve input data path
    if data_path is None or str(data_path).strip() == "":
        data_path = data_paths.get("data_path", "")

    if not data_path or not os.path.exists(data_path):
        raw_json_path = data_paths.get(
            "raw_json",
            os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reddit_post_data.json"),
        )
        if os.path.exists(raw_json_path):
            logger.warning(
                "Processed data path not found (%s). Falling back to raw JSON: %s",
                data_path,
                raw_json_path,
            )
            data_path = raw_json_path
        else:
            logger.warning(
                "Processed data path not found (%s). Raw JSON path missing: %s",
                data_path,
                raw_json_path,
            )

    # Artifact paths
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", data_paths["data_artifacts_dir"])
    os.makedirs(artifacts_dir, exist_ok=True)
    processed_dir = os.path.join(os.path.dirname(__file__), "..", data_paths["processed_data_dir"])
    os.makedirs(processed_dir, exist_ok=True)
    X_train_path = os.path.join(artifacts_dir, "X_train.csv")
    X_test_path = os.path.join(artifacts_dir, "X_test.csv")
    Y_train_path = os.path.join(artifacts_dir, "Y_train.csv")
    Y_test_path = os.path.join(artifacts_dir, "Y_test.csv")

    if (
        os.path.exists(X_train_path)
        and os.path.exists(X_test_path)
        and os.path.exists(Y_train_path)
        and os.path.exists(Y_test_path)
    ):
        logger.info("Found existing artifact files. Loading artifacts...")
        try:
            X_train = pd.read_csv(X_train_path).values
            X_test = pd.read_csv(X_test_path).values
            Y_train = pd.read_csv(Y_train_path).values.ravel()
            Y_test = pd.read_csv(Y_test_path).values.ravel()
            logger.info(
                "Artifacts loaded: X_train=%s, X_test=%s, Y_train=%s, Y_test=%s",
                X_train.shape,
                X_test.shape,
                Y_train.shape,
                Y_test.shape,
            )
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logger.exception("Failed to load artifact files: %s", e)

    logger.info("Preparing openness direction dataset...")

    is_json = str(data_path).lower().endswith(".json")
    csv_usecols = None
    embedding_columns = [raw_embedding_col]
    if not is_json:
        try:
            csv_column_map = DataIngestorCSV.get_column_map(data_path)
            csv_columns = set(csv_column_map)
            weekly_min_columns = {author_col, week_col, obs_col, emb_col}
            if weekly_min_columns.issubset(csv_columns):
                csv_usecols = [
                    csv_column_map[col]
                    for col in [author_col, week_col, obs_col, emb_col, std_col]
                    if col and col in csv_columns
                ]
                embedding_columns = []
                logger.info(
                    "Weekly dataset detected. Reading only required columns: %s",
                    csv_usecols,
                )
        except Exception as e:
            logger.warning("Failed to inspect CSV header for optimized load: %s", e)
    ingestor = DataIngestorJson() if is_json else DataIngestorCSV(
        embedding_columns=embedding_columns,
        usecols=csv_usecols,
    )
    try:
        logger.info("Data ingestion starting.")
        df = ingestor.ingest(data_path)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # df = pd.DataFrame(data)
        logger.info(
            "Data ingested successfully. Items: %s",
            (df.shape[0], df.shape[1]),
        )
    except Exception as e:
        logger.exception("Data ingestion failed: %s", e)
        raise
    _log_df_state(df, "ingest_raw")

    has_weekly_traits = obs_col in df.columns
    has_weekly_embeddings = emb_col in df.columns
    is_weekly = has_weekly_traits and has_weekly_embeddings

    attribute_concat = AttributeConcatenation(
        title_col=title_col,
        selftext_col=selftext_col,
    )
    drop_attributes = DropAttributes(required_attributes)
    text_clean = CleanText()
    replace_emoji = ReplaceEmoji()

    if processed_text_col not in df.columns:
        if is_weekly:
            logger.info(
                "Processed text missing, but weekly data detected. Skipping text preprocessing."
            )
        else:
            try:
                logger.info("Preprocessing starting.")
                df = attribute_concat.preprocess(df)
                logger.info("Post concatenation complete. Items: %s", len(df))
                _log_df_state(df, "attribute_concatenation")
                df = drop_attributes.preprocess(df)
                logger.info("DropAttributes preprocessing complete. Items: %s", len(df))
                _log_df_state(df, "drop_attributes")
                df = text_clean.preprocess(df)
                logger.info("Text clean preprocessing complete. Items: %s", len(df))
                _log_df_state(df, "clean_text")
                df = replace_emoji.preprocess(df)
                logger.info("Emoji clean preprocessing complete. Items: %s", len(df))
                _log_df_state(df, "replace_emoji")
            except Exception as e:
                logger.exception("Data preprocessing failed: %s", e)
                raise

    has_trait_cols = all(trait in df.columns for trait in traits)
    if not has_trait_cols and not is_weekly:
        if scores_col not in df.columns:
            raise ValueError(
                f"Missing '{scores_col}' column required to parse trait scores."
            )
        data_parser = TraitScoreParser(scores_column=scores_col)
        try:
            logger.info("Trait scores parsing starting...")
            df = data_parser.parse(df)
            _log_df_state(df, "trait_score_parse")
        except Exception as e:
            logger.exception("Trait scores parsing failed: %s", e)
            raise
    else:
        logger.info("Trait scores parsing skipped (traits already present or weekly data).")

    if raw_embedding_col in df.columns and emb_col not in df.columns:
        emb_parser = TextEmbeddingParser(
            embedding_column=raw_embedding_col,
            output_column=raw_embedding_col,
            validate_dims=True,
        )
        try:
            logger.info("Raw text embeddings parsing starting...")
            df = emb_parser.parse(df)
            _log_df_state(df, "embedding_parse_raw")
        except Exception as e:
            logger.exception("Raw text embeddings parsing failed: %s", e)
            raise
    
    # If data isn't already weekly, aggregate to user-week
    if not is_weekly:
        trait_aggregator = TraitWeeklyAggregator(
            trait_columns=traits,
            created_date_col=created_date_col,
            week_col=week_col,
        )
        embedding_aggregator = EmbeddingWeeklyAggregator(
            embedding_column=raw_embedding_col,
            output_column=emb_col,
            created_date_col=created_date_col,
            week_col=week_col,
        )

        try:
            logger.info("Weekly trait aggregation starting...")
            weekly_traits = trait_aggregator.aggregate(df, groupby_cols=[author_col, week_col])
            logger.info("Weekly traits aggregated. Shape: %s", weekly_traits.shape)
            _log_df_state(weekly_traits, "weekly_traits")
        except Exception as e:
            logger.exception("Weekly trait aggregation failed: %s", e)
            raise

        if raw_embedding_col not in df.columns:
            if processed_text_col in df.columns:
                logger.info("Text embeddings missing. Generating embeddings...")
                embedder = VectorEmbedding()
                df = embedder.generate_embeddings(
                    df,
                    text_column=processed_text_col,
                    output_column=raw_embedding_col,
                )
            else:
                raise ValueError(
                    f"No '{raw_embedding_col}' or '{processed_text_col}' available for "
                    "embedding generation."
                )

        try:
            logger.info("Weekly embedding aggregation starting...")
            weekly_embeddings = embedding_aggregator.aggregate(
                df, groupby_cols=[author_col, week_col]
            )
            logger.info("Weekly embeddings aggregated. Shape: %s", weekly_embeddings.shape)
            _log_df_state(weekly_embeddings, "weekly_embeddings")
        except Exception as e:
            logger.exception("Weekly embedding aggregation failed: %s", e)
            raise

        weekly_traits_all = None
        try:
            if analysis_traits:
                missing_traits = [t for t in analysis_traits if t not in df.columns]
                if missing_traits:
                    logger.warning(
                        "All-traits weekly aggregation skipped. Missing traits: %s",
                        missing_traits,
                    )
                else:
                    logger.info("Weekly all-traits aggregation starting...")
                    weekly_traits_all = TraitWeeklyAggregator(
                        trait_columns=analysis_traits,
                        created_date_col=created_date_col,
                        week_col=week_col,
                    ).aggregate(df, groupby_cols=[author_col, week_col])
                    logger.info(
                        "Weekly all-traits aggregated. Shape: %s",
                        weekly_traits_all.shape,
                    )
                    _log_df_state(weekly_traits_all, "weekly_traits_all5")

                    # weekly_traits_all_path = os.path.join(
                    #     artifacts_dir, "weekly_traits_all5.csv"
                    # )
                    # weekly_traits_all_out = weekly_traits_all.copy()
                    # if week_col in weekly_traits_all_out.columns:
                    #     weekly_traits_all_out[week_col] = weekly_traits_all_out[week_col].astype(
                    #         str
                    #     )
                    # weekly_traits_all_out.to_csv(weekly_traits_all_path, index=False)
                    # logger.info(
                    #     "Weekly all-traits dataset saved to %s",
                    #     weekly_traits_all_path,
                    # )
        except Exception as e:
            logger.exception("Weekly all-traits aggregation failed: %s", e)

        processed_posts_path = os.path.join(processed_dir, "weekly_processed_posts.csv")
        try:
            df_post = df.copy()
            if week_col not in df_post.columns and created_date_col in df_post.columns:
                df_post[created_date_col] = pd.to_datetime(df_post[created_date_col])
                df_post[week_col] = df_post[created_date_col].dt.to_period("W")
            df_post = df_post.merge(weekly_traits, on=[author_col, week_col], how="left")
            df_post = df_post.merge(weekly_embeddings, on=[author_col, week_col], how="left")
            if week_col in df_post.columns:
                df_post[week_col] = df_post[week_col].astype(str)
            df_post.to_csv(processed_posts_path, index=False)
            logger.info("Weekly processed posts saved to %s", processed_posts_path)
        except Exception as e:
            logger.exception("Failed to save weekly processed posts: %s", e)

        if weekly_traits_all is not None:
            processed_posts_all_path = os.path.join(
                processed_dir, "weekly_processed_posts_all5.csv"
            )
            try:
                df_post_all = df.copy()
                if week_col not in df_post_all.columns and created_date_col in df_post_all.columns:
                    df_post_all[created_date_col] = pd.to_datetime(df_post_all[created_date_col])
                    df_post_all[week_col] = df_post_all[created_date_col].dt.to_period("W")
                df_post_all = df_post_all.merge(
                    weekly_traits_all, on=[author_col, week_col], how="left"
                )
                df_post_all = df_post_all.merge(
                    weekly_embeddings, on=[author_col, week_col], how="left"
                )
                if week_col in df_post_all.columns:
                    df_post_all[week_col] = df_post_all[week_col].astype(str)
                df_post_all.to_csv(processed_posts_all_path, index=False)
                logger.info(
                    "Weekly processed posts (all traits) saved to %s",
                    processed_posts_all_path,
                )
            except Exception as e:
                logger.exception(
                    "Failed to save weekly processed posts (all traits): %s", e
                )

        df = weekly_traits.merge(weekly_embeddings, on=[author_col, week_col], how="inner")
        _log_df_state(df, "weekly_merge")

    # Build openness working dataframe
    optional_cols = [
        c
        for c in ["time_gap_weeks", "log_time_gap_weeks"]
        if c in df.columns
    ]
    if std_col and std_col in df.columns:
        optional_cols.append(std_col)
    keep_cols = [author_col, week_col, obs_col, emb_col] + optional_cols
    df_o = df[keep_cols].copy()
    _log_df_state(df_o, "openness_subset")

    # Parse aggregated embeddings to vectors
    emb_parser = TextEmbeddingParser(
        embedding_column=emb_col,
        output_column="emb_vec",
        validate_dims=True,
    )
    df_o = emb_parser.parse(df_o)
    _log_df_state(df_o, "embedding_parse")

    # Week parsing and time-gap features
    gap_calc = GapWeekCalculator(author_col=author_col, week_col=week_col, time_col=time_sort_col)
    df_o = gap_calc.calculate(df_o)
    _log_df_state(df_o, "time_gap_features")

    # Option A label components
    delta_creator = FutureDeltaTargetCreator(
        obs_column=obs_col,
        past_window=k_past,
        future_window=h_future,
        time_column=time_sort_col,
    )
    df_o = delta_creator.create(df_o, groupby_col=author_col)
    _log_df_state(df_o, "future_delta_targets")

    # Remove rows without sufficient context
    before_rows = df_o.shape[0]
    df_o = df_o.dropna(subset=["past_mean", "future_mean", "delta_future"]).copy()
    after_rows = df_o.shape[0]
    logger.info("Context filter: rows before=%s after=%s", before_rows, after_rows)
    _log_df_state(df_o, "context_filtered")

    # Author-wise longitudinal train/test split
    splitter = AuthorTimeSplitStrategy(
        test_size=test_size,
        author_col=author_col,
        time_col=time_sort_col,
        drop_small_groups=True,
        min_rows=2,
        use_ceil_for_test=False,
    )
    df_o = splitter.add_split_column(df_o, column="is_train")
    _log_df_state(df_o, "author_forward_split")

    # Tau and direction labels from training only
    labeler = DirectionLabeler(
        delta_column="delta_future",
        label_column=label_col,
        tau_quantile=tau_quantile,
        train_mask_column="is_train",
    )
    df_o = labeler.fit_transform(df_o)
    logger.info("Tau used for labels: %.6f", labeler.tau_)
    _log_df_state(df_o, "direction_labels")

    # Feature construction
    if "openness_lag1" not in df_o.columns:
        df_o["openness_lag1"] = df_o[obs_col]
    _log_df_state(df_o, "openness_lag1_base")

    rolling_creator = RollingStatsFeatureCreator(
        value_column=obs_col,
        group_column=author_col,
        window=4,
        min_periods=2,
        mean_column="openness_roll4",
        std_column="openness_roll4_std",
    )
    df_o = rolling_creator.create(df_o)
    _log_df_state(df_o, "rolling_stats")

    numeric_features = [
        "time_gap_weeks",
        "log_time_gap_weeks",
        "openness_lag1",
        "openness_roll4",
        "openness_roll4_std",
    ]
    numeric_features_cp = ["openness_lag1", "openness_roll4", "openness_roll4_std"]

    if std_col:
        if std_col not in df_o.columns:
            raise ValueError(
                f"Expected std feature column '{std_col}' not found in data."
            )
        numeric_features.append(std_col)

    shift_creator = FeatureShiftCreator(
        feature_columns=numeric_features_cp,
        group_column=author_col,
        lag=1,
    )
    df_o = shift_creator.create(df_o)
    df_o["emb_vec_shift"] = df_o.groupby(author_col)["emb_vec"].shift(1)
    _log_df_state(df_o, "feature_shift")

    required = numeric_features + ["emb_vec_shift", label_col, "is_train"]
    df_o = df_o.sort_values([author_col, time_sort_col]).reset_index(drop=True)
    model_df = df_o.dropna(subset=required).copy()
    _log_df_state(model_df, "model_df_ready")

    if model_df.empty:
        raise ValueError("No rows left after preprocessing. Check input data and parameters.")

    X_num = model_df[numeric_features].to_numpy(dtype=float)
    X_emb = np.vstack(
        model_df["emb_vec_shift"].apply(lambda v: np.asarray(v, dtype=float)).to_numpy()
    )
    X = np.hstack([X_num, X_emb])
    y = model_df[label_col].to_numpy()
    train_mask = model_df["is_train"].to_numpy()

    X_train, X_test = X[train_mask], X[~train_mask]
    Y_train, Y_test = y[train_mask], y[~train_mask]

    emb_dim = X_emb.shape[1]
    emb_feature_names = [f"emb_vec_shift_{i}" for i in range(emb_dim)]
    X_feature_names = numeric_features + emb_feature_names

    logger.info("Train/Test split complete.")
    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("Y_train shape: %s", Y_train.shape)
    logger.info("Y_test shape: %s", Y_test.shape)

    pd.DataFrame(X_train, columns=X_feature_names).to_csv(X_train_path, index=False)
    pd.DataFrame(X_test, columns=X_feature_names).to_csv(X_test_path, index=False)
    pd.DataFrame(Y_train, columns=[label_col]).to_csv(Y_train_path, index=False)
    pd.DataFrame(Y_test, columns=[label_col]).to_csv(Y_test_path, index=False)
    logger.info("Splits saved as CSV files.")

    try:
        env_config = get_environment_config(domain=DOMAIN, trait=TRAIT)
        mlflow_config = get_mlflow_config()
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
            "data_pipeline",
            {
                "week_window": str(env_config.get("week_window", 1)),
                "dataset_version": env_config.get("dataset_version", "v1.0"),
                "target_type": "openness_direction",
                "num_targets": "1",
            },
        )
        run_name = (
            f"{mlflow_config.get('run_name_prefix', 'run')}_data_"
            f"{env_config.get('week_window', 1)}w"
        )
        mlflow_tracker.start_run(run_name=run_name, tags=run_tags)
        label_counts = model_df[label_col].value_counts(dropna=False).to_dict()
        train_label_counts = (
            model_df.loc[model_df["is_train"], label_col].value_counts(dropna=False).to_dict()
        )
        test_label_counts = (
            model_df.loc[~model_df["is_train"], label_col].value_counts(dropna=False).to_dict()
        )
        dataset_info = {
            "total_rows": int(model_df.shape[0]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "num_features": int(X.shape[1]),
            "test_size": float(test_size),
            "random_state": 42,
            "feature_names": X_feature_names,
            "tau": float(labeler.tau_) if labeler.tau_ is not None else None,
            "tau_quantile": float(tau_quantile),
            "label_counts": label_counts,
            "label_counts_train": train_label_counts,
            "label_counts_test": test_label_counts,
            "authors_train": int(model_df.loc[model_df["is_train"], author_col].nunique()),
            "authors_test": int(model_df.loc[~model_df["is_train"], author_col].nunique()),
            "preprocessing_steps": [
                "ingest",
                "text_preprocess",
                "trait_parse",
                "weekly_aggregate",
                "embeddings",
                "gap_features",
                "direction_labels",
                "train_test_split",
            ],
        }
        mlflow_tracker.log_data_pipeline_metrics(dataset_info)
        metadata_path = data_paths.get(
            "data_pipeline_metadata",
            os.path.join(artifacts_dir, "data_pipeline_metadata.json"),
        )
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, default=str)
        mlflow.log_artifact(metadata_path, "data_pipeline")
        mlflow_tracker.end_run()
    except Exception as e:
        logger.warning("MLflow data pipeline logging failed: %s", e)

    return X_train, X_test, Y_train, Y_test

def main() -> None:
    """Run the module entry point."""
    data_pipeline()

if __name__ == "__main__":
    main()




