# Databricks notebook source
# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------
# MAGIC %run ../utils

# COMMAND ----------

import pandas as pd
from sklearn.metrics import silhouette_score

# COMMAND ----------

def get_prediction(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    spark_df = read_current_table(
        spark=spark,
        table_base=f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}",
    )

    return spark_df

# COMMAND ----------

RANDOM_SEED = 42
KMEANS_TRY_RANGE = [3, 4, 5, 6, 7, 8, 9, 10]

# COMMAND ----------

def evaluate_analysis(analysis: str):
    spark_df = get_prediction(analysis)
    predict_df = spark_df.toPandas()

    pca_cols = [col for col in predict_df.columns if col.startswith("pca_")]
    X_pca = predict_df[pca_cols].to_numpy()
    cluster_ids = predict_df["cluster_id"].str.replace("cluster_", "").astype(int).to_numpy()

    silhouette = silhouette_score(X_pca, cluster_ids, metric="euclidean")

    return silhouette

# COMMAND ----------

def evaluate_all_analysis():
    result_dict = {}
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        silhouette = evaluate_analysis(analysis)
        result_dict[analysis] = silhouette

    return result_dict


# COMMAND ----------

def save_evaluated_score(analysis_id, analysis_name, kmeans_k, pca_k, sil_score):
    sil_score = float(sil_score)
    
    df = spark.createDataFrame(
        [(analysis_id, analysis_name, kmeans_k, pca_k, sil_score)],
        ["analysis_id", "analysis_name", "kmeans_k", "pca_k", "sil_score"]
    ).withColumn("created_at", F.current_timestamp())

    try:
        current_df = read_current_table(spark=spark, table_base=T_GOLD_CUSTOMER_EVAL_SCORE)
        combined_df = current_df.unionByName(df, allowMissingColumns=True)
    except Exception:
        combined_df = df

    publish_table(
        spark=spark,
        df=combined_df,
        table_base=T_GOLD_CUSTOMER_EVAL_SCORE,
        analysis_name=analysis_name,
        current_mode="table",
    )

# COMMAND ----------

def evaluate_kmeans_k_list(analysis:str, kmeans_try_list=[]):
    analysis_name = ANALYSIS_NAMES[analysis]

    for kmeans_k in kmeans_try_list:
        ml_parameters = {
            "kmeans_k": kmeans_k,
            "seed": RANDOM_SEED
        }
        print(f"=== Testing K={kmeans_k} ===")

        pca_k_by_analysis = train_predict_analysis(analysis, ml_parameters)
        silhouette = evaluate_analysis(analysis)

        save_evaluated_score(
            analysis_id=analysis,
            analysis_name=analysis_name,
            kmeans_k=kmeans_k,
            pca_k=pca_k_by_analysis,
            sil_score=silhouette
        )
