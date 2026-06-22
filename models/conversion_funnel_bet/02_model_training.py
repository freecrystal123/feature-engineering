# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 - Conversion funnel model training (RandomForest + SHAP)

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_bet

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------
# MAGIC %run ../../core/uc_feature_store

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("candidate_alias", DEFAULT_CANDIDATE_ALIAS)

# COMMAND ----------

import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

analysis_id = dbutils.widgets.get("analysis_id").strip()
candidate_alias = dbutils.widgets.get("candidate_alias").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def create_model_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=20,
                    random_state=int(DEFAULT_RANDOM_SEED),
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_pr_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _build_train_test_split(model_df: pd.DataFrame):
    model_df = model_df.copy()
    model_df["deposit_month"] = pd.to_datetime(model_df["deposit_month"], errors="coerce")

    month_values = sorted([m for m in model_df["deposit_month"].dropna().unique()])
    if len(month_values) >= 3:
        n_test_months = max(1, int(round(len(month_values) * 0.2)))
        n_test_months = min(n_test_months, len(month_values) - 1)
        test_months = set(month_values[-n_test_months:])
        test_mask = model_df["deposit_month"].isin(test_months)
        train_mask = ~test_mask
        if train_mask.sum() >= 100 and test_mask.sum() >= 50:
            holdout = ",".join(pd.to_datetime(list(test_months)).strftime("%Y-%m-%d").tolist())
            return (
                model_df.loc[train_mask].copy(),
                model_df.loc[test_mask].copy(),
                "temporal_month_holdout",
                holdout,
            )

    stratify_target = model_df["label_wager_success_30d"] if model_df["label_wager_success_30d"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=int(DEFAULT_RANDOM_SEED),
        stratify=stratify_target,
    )
    return train_df.copy(), test_df.copy(), "random_stratified", "n/a"


def _compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
    }


def _compute_shap_global_importance(model_pipeline, X_frame: pd.DataFrame):
    try:
        import shap
    except Exception as exc:
        raise ImportError(
            "The 'shap' package is required for this analysis. Install shap in the cluster before running training."
        ) from exc

    clf = model_pipeline.named_steps["clf"]
    imputer = model_pipeline.named_steps["imputer"]

    shap_sample_size = min(5000, len(X_frame))
    if shap_sample_size <= 0:
        raise ValueError("Unable to compute SHAP: empty training sample.")

    X_sample = X_frame.sample(n=shap_sample_size, random_state=int(DEFAULT_RANDOM_SEED))
    X_imputed = imputer.transform(X_sample)
    transformed_feature_names = list(X_sample.columns)
    if hasattr(imputer, "get_feature_names_out"):
        try:
            transformed_feature_names = list(imputer.get_feature_names_out(X_sample.columns))
        except Exception:
            transformed_feature_names = list(X_sample.columns)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_imputed)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            shap_matrix = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
        else:
            shap_matrix = arr

    # Defensive alignment: some sklearn versions can drop all-missing columns in
    # SimpleImputer, changing transformed width vs original feature list.
    n_shap_cols = int(shap_matrix.shape[1])
    if len(transformed_feature_names) != n_shap_cols:
        transformed_feature_names = transformed_feature_names[:n_shap_cols]
        if len(transformed_feature_names) < n_shap_cols:
            transformed_feature_names.extend([f"feature_{i}" for i in range(len(transformed_feature_names), n_shap_cols)])

    shap_df = pd.DataFrame(
        {
            "feature_name": transformed_feature_names,
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            "mean_shap": np.mean(shap_matrix, axis=0),
            "sample_size": int(shap_sample_size),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df["feature_rank"] = np.arange(1, len(shap_df) + 1)
    return shap_df


def train_and_register(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}"
    primary_table = snapshot_table_name(table_base)
    entity_source_table = current_table_name(table_base)

    table_df = spark.table(primary_table)
    feature_cols = get_funnel_feature_columns(
        table_df,
        analysis_id=analysis,
        analysis_name=analysis_name,
    )
    if not feature_cols:
        raise ValueError(f"No numeric feature columns inferred from table '{primary_table}'.")

    ts_result = create_training_set_from_feature_tables(
        spark=spark,
        primary_table=primary_table,
        entity_source_table=entity_source_table,
        entity_keys="unique_user_id",
        primary_feature_names=feature_cols,
        secondary_lookups=[],
        label="label_wager_success_30d",
        exclude_columns=["unique_user_id", "run_ts"],
        entity_timestamp_col="run_ts",
        entity_extra_cols=None,
    )

    training_set = ts_result["training_set"]
    training_source_df = spark.table(entity_source_table)
    if "observation_complete" not in training_source_df.columns:
        raise ValueError(
            f"Training source '{entity_source_table}' is missing observation_complete."
        )
    train_pdf = (
        training_source_df
        .filter(F.col("observation_complete") == F.lit(True))
        .select(*(feature_cols + ["label_wager_success_30d", "deposit_month"]))
        .toPandas()
    )

    if train_pdf.empty:
        raise ValueError(
            "No rows available for training after applying minimum observation window "
            f"({int(LABEL_WINDOW_DAYS)} days since first deposit)."
        )
    if train_pdf["label_wager_success_30d"].nunique() < 2:
        raise ValueError("Training label has a single class after minimum observation window filtering.")

    # Drop features that are fully null in this training extract (common when old
    # columns remain in merged table schemas but are no longer populated).
    non_empty_feature_cols = []
    dropped_all_missing = []
    for col in feature_cols:
        series = pd.to_numeric(train_pdf[col], errors="coerce")
        if series.notna().any():
            non_empty_feature_cols.append(col)
        else:
            dropped_all_missing.append(col)
    feature_cols = non_empty_feature_cols

    if dropped_all_missing:
        print(
            json.dumps(
                {
                    "dropped_all_missing_features": dropped_all_missing,
                    "n_dropped": len(dropped_all_missing),
                }
            )
        )
    if not feature_cols:
        raise ValueError("No usable feature columns after dropping all-missing features.")

    train_split_pdf, test_split_pdf, split_strategy, holdout_months = _build_train_test_split(train_pdf)

    X_train = prepare_funnel_model_input(train_split_pdf, feature_cols)
    y_train = train_split_pdf["label_wager_success_30d"].astype("int64")

    X_test = prepare_funnel_model_input(test_split_pdf, feature_cols)
    y_test = test_split_pdf["label_wager_success_30d"].astype("int64")

    model = create_model_pipeline()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_score = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_score = model.predict_proba(X_test)[:, 1]

    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_score)
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_score)

    shap_pdf = _compute_shap_global_importance(model, X_train)

    model_name = build_model_name(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
    )

    with mlflow.start_run(run_name=f"train_{analysis_name}"):
        mlflow.log_params(
            {
                "subject": SUBJECT_NAME,
                "analysis_id": analysis,
                "analysis_name": analysis_name,
                "write_catalog": WRITE_CATALOG,
                "n_rows_total": int(len(train_pdf)),
                "n_rows_train": int(len(train_split_pdf)),
                "n_rows_test": int(len(test_split_pdf)),
                "n_features": int(X_train.shape[1]),
                "candidate_alias": candidate_alias,
                "split_strategy": split_strategy,
                "holdout_months": holdout_months,
                "cohort_lookback_months": int(COHORT_LOOKBACK_MONTHS),
                "label_window_days": int(LABEL_WINDOW_DAYS),
            }
        )
        mlflow.log_text("\n".join(feature_cols), "features.txt")
        mlflow.log_metrics({f"train_{k}": float(v) for k, v in train_metrics.items() if pd.notna(v)})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items() if pd.notna(v)})

        top_shap_text = "\n".join(
            [f"{r.feature_rank}. {r.feature_name}: {r.mean_abs_shap:.6f}" for r in shap_pdf.head(40).itertuples(index=False)]
        )
        mlflow.log_text(top_shap_text, "shap_global_top40.txt")

        log_model_with_feature_lineage(
            fe_client=get_feature_engineering_client(),
            model=model,
            training_set=training_set,
            registered_model_name=model_name,
            flavor=mlflow.sklearn,
            artifact_path="model",
            signature=None,
            input_example=None,
            extra_kwargs={"skops_trusted_types": ["numpy.dtype"]},
        )

        run_id = mlflow.active_run().info.run_id
        candidate = register_candidate(
            model_name=model_name,
            run_id=run_id,
            alias=candidate_alias,
            tags={
                "analysis": analysis_name,
                "governance_status": "candidate",
                "subject": SUBJECT_NAME,
            },
        )

        eval_rows = [
            {
                "split_name": "train",
                "split_strategy": split_strategy,
                "holdout_months": holdout_months,
                "n_rows": int(len(train_split_pdf)),
                "positive_rate": float(y_train.mean()),
                **train_metrics,
            },
            {
                "split_name": "test",
                "split_strategy": split_strategy,
                "holdout_months": holdout_months,
                "n_rows": int(len(test_split_pdf)),
                "positive_rate": float(y_test.mean()),
                **test_metrics,
            },
        ]

        eval_df = spark.createDataFrame(pd.DataFrame(eval_rows))
        eval_publish = publish_table(
            spark=spark,
            df=eval_df.withColumn("created_at", F.current_timestamp()),
            table_base=f"{T_FUNNEL_EVAL_PREFIX}_{analysis_name}",
            analysis_name=analysis_name,
            run_id=run_id,
            current_mode="view",
            primary_keys=["split_name", "run_ts"],
            timeseries_column="run_ts",
        )

        shap_df = spark.createDataFrame(shap_pdf)
        shap_publish = publish_table(
            spark=spark,
            df=shap_df.withColumn("created_at", F.current_timestamp()),
            table_base=f"{T_FUNNEL_SHAP_PREFIX}_{analysis_name}",
            analysis_name=analysis_name,
            run_id=run_id,
            current_mode="view",
            primary_keys=["feature_name", "run_ts"],
            timeseries_column="run_ts",
        )

        return {
            "analysis": analysis,
            "analysis_name": analysis_name,
            "model_name": model_name,
            "candidate_alias": candidate_alias,
            "candidate_version": candidate["version"],
            "auto_promoted": False,
            "champion_alias": DEFAULT_CHAMPION_ALIAS,
            "split_strategy": split_strategy,
            "holdout_months": holdout_months,
            "metrics": {
                "train": train_metrics,
                "test": test_metrics,
            },
            "eval_current_table": eval_publish["current_table"],
            "shap_current_table": shap_publish["current_table"],
        }


results = [train_and_register(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
