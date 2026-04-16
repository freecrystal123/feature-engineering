# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 - model prediction (alias-based)

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------
# MAGIC %run ../../core/uc_feature_store

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
    feature_table = snapshot_table_name(table_base)
    entity_source_table = current_table_name(table_base)
    table_df = spark.table(feature_table)
    if "unique_user_id" not in table_df.columns:
        raise ValueError(f"unique_user_id doesn't exist in feature table '{feature_table}'")

    metadata_cols = {"run_id", "run_ts", "ref_date", "analysis_name", "created_at"}
    numeric_cols = [
        f.name
        for f in table_df.schema.fields
        if f.name not in (metadata_cols | {"unique_user_id"})
        and f.dataType.typeName() in {"byte", "short", "integer", "long", "float", "double", "decimal"}
    ]
    if not numeric_cols:
        raise ValueError(f"No numeric feature columns found in table '{feature_table}'.")

    ts_result = create_training_set_from_feature_tables(
        spark=spark,
        primary_table=feature_table,
        entity_source_table=entity_source_table,
        entity_keys="unique_user_id",
        primary_feature_names=numeric_cols,
        secondary_lookups=[],
        label=None,
        exclude_columns=[],
        entity_timestamp_col="run_ts",
    )
    features = (
        ts_result["training_set"]
        .load_df()
        .orderBy("unique_user_id")
        .toPandas()
    )
    user_ids = features["unique_user_id"].copy()
    X = (
        features[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )
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
    cluster_base = f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}"

    # Preserve existing PCA names/descriptions in daily prediction runs.
    try:
        existing_pca_meta = (
            read_current_table(spark=spark, table_base=pca_base)
            .select("pca_id", "pca_component_name", "pca_component_description")
            .dropDuplicates(["pca_id"])
        )
        pca_spark = (
            pca_spark.alias("n")
            .join(existing_pca_meta.alias("m"), on="pca_id", how="left")
            .select(
                "pca_id",
                *[c for c in pca_spark.columns if c not in {"pca_id"}],
                "pca_component_name",
                "pca_component_description",
            )
        )
    except Exception:
        # First run may not have prior metadata yet.
        pass

    pca_publish = publish_table(
        spark=spark,
        df=pca_spark,
        table_base=pca_base,
        analysis_name=analysis_name,
        current_mode="view",
    )
    pred_publish = publish_table(
        spark=spark,
        df=pred_spark,
        table_base=pred_base,
        analysis_name=analysis_name,
        run_id=pca_publish["run_id"],
        current_mode="view",
    )

    # Keep cluster naming stable in daily prediction:
    # update only population by cluster_id using existing cluster naming table.
    cluster_publish = None
    try:
        existing_clusters = read_current_table(spark=spark, table_base=cluster_base)
        if existing_clusters.limit(1).count() == 0:
            cluster_snapshot = snapshot_table_name(cluster_base)
            if spark.catalog.tableExists(cluster_snapshot):
                latest_run = (
                    spark.table(cluster_snapshot)
                    .groupBy("run_id")
                    .agg(F.max("run_ts").alias("max_run_ts"))
                    .orderBy(F.col("max_run_ts").desc())
                    .select("run_id")
                    .limit(1)
                    .collect()
                )
                if latest_run:
                    existing_clusters = spark.table(cluster_snapshot).filter(
                        F.col("run_id") == F.lit(latest_run[0]["run_id"])
                    )

        fresh_counts = (
            pred_spark.groupBy("cluster_id")
            .agg(F.count("*").alias("num_users"))
        )
        metadata_cols = {"run_id", "run_ts", "ref_date", "analysis_name"}
        cluster_cols = [c for c in existing_clusters.columns if c not in metadata_cols]
        updated_clusters = (
            existing_clusters.select(*cluster_cols).drop("num_users")
            .join(fresh_counts, on="cluster_id", how="left")
            .withColumn("num_users", F.coalesce(F.col("num_users"), F.lit(0)))
        )
        cluster_publish = publish_table(
            spark=spark,
            df=updated_clusters.withColumn("created_at", F.current_timestamp()),
            table_base=cluster_base,
            analysis_name=analysis_name,
            run_id=pca_publish["run_id"],
            current_mode="view",
        )
    except Exception:
        cluster_publish = None

    return {
        "analysis": analysis,
        "analysis_name": analysis_name,
        "model_alias": model_alias,
        "pca_current": pca_publish["current_table"],
        "pred_current": pred_publish["current_table"],
        "cluster_current": cluster_publish["current_table"] if cluster_publish else None,
        "run_id": pca_publish["run_id"],
    }


results = [predict_analysis(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
