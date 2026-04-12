# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 - model prediction (alias-based)

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------
# MAGIC %run ../utils

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("model_alias", "champion")

# COMMAND ----------

import json
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

analysis_id = dbutils.widgets.get("analysis_id").strip()
model_alias = dbutils.widgets.get("model_alias").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def get_X_for_prediction(analysis: str):
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


def predict_with_pipeline(pipeline, X: pd.DataFrame, user_ids: pd.Series):
    X_imputed = pipeline["imputer_zero"].transform(X)
    X_scaled = pipeline["scaler"].transform(X_imputed)
    X_pca = pipeline["pca"].transform(X_scaled)

    cluster_ids = pipeline["kmeans"].predict(X_pca)
    real_pca_k = int(pipeline["pca"].n_components_)
    explained = pipeline["pca"].explained_variance_ratio_[:real_pca_k]

    pca_components = pd.DataFrame(
        pipeline["pca"].components_,
        columns=X.columns.tolist(),
    )
    pca_components["pca_id"] = [f"pca_{i+1}" for i in range(real_pca_k)]
    pca_components["explained_variance_ratio"] = explained
    pca_components["explained_variance_ratio_cum"] = np.cumsum(explained)

    predict_df = pd.DataFrame(
        X_pca,
        columns=[f"pca_{i+1}" for i in range(real_pca_k)],
    )
    predict_df.insert(0, "unique_user_id", user_ids.values)
    predict_df["cluster_id"] = [f"cluster_{c}" for c in cluster_ids]

    return pca_components, predict_df


def predict_analysis(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    model = load_model(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
        alias=model_alias,
    )
    X, user_ids = get_X_for_prediction(analysis)
    pca_components, predict_df = predict_with_pipeline(model, X, user_ids)

    pca_spark = spark.createDataFrame(pca_components).withColumn("created_at", F.current_timestamp())
    pred_spark = spark.createDataFrame(predict_df).withColumn("created_at", F.current_timestamp())

    pca_base = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"
    pred_base = f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}"

    pca_publish = publish_table(
        spark=spark,
        df=pca_spark,
        table_base=pca_base,
        analysis_name=analysis_name,
        current_mode="table",
    )
    pred_publish = publish_table(
        spark=spark,
        df=pred_spark,
        table_base=pred_base,
        analysis_name=analysis_name,
        run_id=pca_publish["run_id"],
        current_mode="table",
    )

    return {
        "analysis": analysis,
        "analysis_name": analysis_name,
        "model_alias": model_alias,
        "pca_current": pca_publish["current_table"],
        "pred_current": pred_publish["current_table"],
        "run_id": pca_publish["run_id"],
    }


results = [predict_analysis(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
