# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 - model training (candidate registration)

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------
# MAGIC %run ../utils

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("kmeans_k", "")
dbutils.widgets.text("seed", "42")
dbutils.widgets.text("candidate_alias", "challenger")
dbutils.widgets.text("required_signature", "true")
dbutils.widgets.text("auto_promote_champion", "false")
dbutils.widgets.text("champion_alias", "champion")

# COMMAND ----------

import json
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlflow.models.signature import infer_signature

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int_or_none(value: str):
    if value is None or value == "":
        return None
    return int(str(value).strip())


analysis_id = dbutils.widgets.get("analysis_id").strip()
kmeans_k_widget = _to_int_or_none(dbutils.widgets.get("kmeans_k"))
seed = int(dbutils.widgets.get("seed"))
candidate_alias = dbutils.widgets.get("candidate_alias").strip()
required_signature = _to_bool(dbutils.widgets.get("required_signature"), default=True)
auto_promote_champion = _to_bool(dbutils.widgets.get("auto_promote_champion"), default=False)
champion_alias = dbutils.widgets.get("champion_alias").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def get_X_for_fitting(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}"
    spark_df = read_current_table(spark=spark, table_base=table_base)

    if "unique_user_id" in spark_df.columns:
        spark_df = spark_df.orderBy("unique_user_id")

    features = spark_df.toPandas()
    if "unique_user_id" not in features.columns:
        raise ValueError("unique_user_id doesn't exist in features table")

    user_ids = features["unique_user_id"].copy()
    numeric_cols = [
        col for col, dtype in features.dtypes.items()
        if dtype in ("int64", "float64") and col != "unique_user_id"
    ]
    X = features[numeric_cols]
    return X, user_ids


def discover_best_pca_k(X, threshold=0.80):
    prep = Pipeline(
        [
            ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=None, random_state=seed)),
        ]
    )
    prep.fit(X)
    explained = prep["pca"].explained_variance_ratio_
    best_pca_k = int((explained.cumsum() < threshold).sum() + 1)
    return best_pca_k


def create_pipeline(pca_k: int, kmeans_k: int):
    return Pipeline(
        [
            ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=pca_k, random_state=seed)),
            ("kmeans", KMeans(n_clusters=kmeans_k, random_state=seed, n_init="auto")),
        ]
    )


def train_and_register(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    model_name = build_model_name(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
    )

    X, _ = get_X_for_fitting(analysis)
    local_kmeans_k = kmeans_k_widget if kmeans_k_widget is not None else int(FIXED_KMEANS_K)
    local_pca_k = discover_best_pca_k(X)
    pipeline = create_pipeline(pca_k=local_pca_k, kmeans_k=local_kmeans_k)
    pipeline.fit(X)

    X_example = X.head(5).copy()
    int_cols = X_example.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns
    if len(int_cols) > 0:
        X_example[int_cols] = X_example[int_cols].astype("float64")
        X_example.loc[X_example.index[0], int_cols[0]] = np.nan

    y_example = pipeline.predict(X_example)
    signature = infer_signature(X_example, y_example)

    with mlflow.start_run(run_name=f"train_{analysis_name}"):
        mlflow.log_params(
            {
                "subject": SUBJECT_NAME,
                "analysis_id": analysis,
                "analysis_name": analysis_name,
                "seed": seed,
                "kmeans_k": local_kmeans_k,
                "pca_k": local_pca_k,
                "n_rows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "write_catalog": WRITE_CATALOG,
                "candidate_alias": candidate_alias,
            }
        )
        mlflow.log_text("\n".join(X.columns.tolist()), "features.txt")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model_name,
            input_example=X_example,
            signature=signature,
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
        validation = validate_candidate(
            model_name=model_name,
            alias=candidate_alias,
            required_signature=required_signature,
            sample_input=X_example,
        )
        if not validation["passed"]:
            raise ValueError(
                f"Candidate validation failed for {model_name} "
                f"(version={candidate['version']}): {validation}"
            )

        if auto_promote_champion:
            promote_alias(
                model_name=model_name,
                source_alias=candidate_alias,
                target_alias=champion_alias,
            )

        return {
            "analysis": analysis,
            "analysis_name": analysis_name,
            "model_name": model_name,
            "candidate_alias": candidate_alias,
            "candidate_version": candidate["version"],
            "validated": bool(validation["passed"]),
            "auto_promoted": bool(auto_promote_champion),
            "champion_alias": champion_alias,
        }


results = [train_and_register(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
