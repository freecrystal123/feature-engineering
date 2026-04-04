# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 - model training

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

# MAGIC %run ./022_evaluation 

# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC drop table uat_gold.machine_learning.pcacomp_online_casino

# COMMAND ----------

import mlflow
import mlflow.sklearn
# mlflow.sklearn.autolog(log_models=True)

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read table for fitting

# COMMAND ----------

def get_X_for_fitting(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    spark_df = spark.table(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}")
                    

    

    # stable order
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the pipeline

# COMMAND ----------


def get_pipeline_steps_list(pca_k: int, seed: int, kmeans_k: int):
    return [
        # ("imputer_mean", SimpleImputer(strategy="mean")),
        ("imputer_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=pca_k, random_state=seed)),
        ("kmeans", KMeans(
            n_clusters=kmeans_k,
            random_state=seed,
            n_init="auto"
        ))
    ]

def create_new_pipeline(pca_k: int, seed: int, kmeans_k: int):
    pipeline_steps_list = get_pipeline_steps_list(pca_k, seed, kmeans_k)
    pipeline = Pipeline(pipeline_steps_list)

    return pipeline


# COMMAND ----------

def discover_best_pca_k(X, seed=RANDOM_SEED, threshold=0.80):

    pipeline_full_steps = get_pipeline_steps_list(pca_k=None, seed=seed, kmeans_k=5)
    pipeline_without_kmeans = Pipeline(pipeline_full_steps[0:-1])

    pipeline_without_kmeans.fit(X)

    # explained variance
    explained = pipeline_without_kmeans["pca"].explained_variance_ratio_

    # ideal PCA_K (ex: 80% variance)
    best_pca_k = int((explained.cumsum() < threshold).sum() + 1)

    return best_pca_k, explained

# COMMAND ----------

from pyspark.sql import functions as F

def get_best_value_kmeans_k(
    analysis: str,
    eps: float = 0.005,
    min_reasonable_k: int = 4,   # <-- interpretability floor
    max_k: int = 50
):
    """
    Picks the smallest k (>= min_reasonable_k) that is close enough to the best silhouette.

    Rule:
      - Consider only k in [min_reasonable_k, max_k]
      - best_sil = max(sil_score) within this range
      - candidates = sil_score >= best_sil - eps
      - return the smallest k among the candidates
      - if no candidates (edge case), return the best silhouette in the range
    """

    eval_df = (
        spark.table(f"{T_GOLD_CUSTOMER_EVAL_SCORE}")
        .filter(F.col("analysis_id") == F.lit(analysis))
        .filter((F.col("kmeans_k") >= F.lit(min_reasonable_k)) & (F.col("kmeans_k") <= F.lit(max_k)))
        .select("kmeans_k", "sil_score", "pca_k")
    )

    rows = eval_df.collect()
    if not rows:
        raise ValueError(f"No evaluation records found for analysis_id={analysis} in k range [{min_reasonable_k}, {max_k}]")

    best_sil = max(float(r["sil_score"]) for r in rows)
    threshold = best_sil - eps

    candidates = [r for r in rows if float(r["sil_score"]) >= threshold]
    if candidates:
        chosen = sorted(candidates, key=lambda r: (int(r["kmeans_k"]), -float(r["sil_score"])))[0]
    else:
        # fallback: best silhouette within range
        chosen = sorted(rows, key=lambda r: -float(r["sil_score"]))[0]

    print(f'Chosen Kmeans_k: {chosen["kmeans_k"]}')
    print(f'With Silhouette score: {chosen["sil_score"]}')

    return chosen["kmeans_k"], chosen["sil_score"], chosen["pca_k"]

# COMMAND ----------

# DBTITLE 1,Untitled
def learn_X_into_pipeline(analysis: str,
                          ml_parameters={"kmeans_k":None, "seed":RANDOM_SEED}):
    X, user_ids = get_X_for_fitting(analysis)

    analysis_name = ANALYSIS_NAMES[analysis]

    if ml_parameters["kmeans_k"] == None:
        best_kmeans_k, best_sil_score , best_pca_k = get_best_value_kmeans_k(analysis)
        ml_parameters["kmeans_k"] = best_kmeans_k
    else:
        best_pca_k = discover_best_pca_k(X, seed=ml_parameters["seed"])[0]
    
    ml_parameters["pca_k"] = best_pca_k

    pipeline = create_new_pipeline(
        pca_k=ml_parameters["pca_k"],
        seed=ml_parameters["seed"],
        kmeans_k=ml_parameters["kmeans_k"]
    )
    
    pipeline.fit(X)

    mlflow.set_experiment("/Users/aneto@mcorp.ae/user_segmentation2")

    X_example = X.head(5)

    int_cols = X_example.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns
    if len(int_cols) > 0:
        X_example[int_cols] = X_example[int_cols].astype("float64")

    for c in int_cols[: min(3, len(int_cols))]:
        X_example.loc[X_example.index[0], c] = np.nan


    y_example = pipeline.predict(X_example)

    signature = infer_signature(X_example, y_example)

    
    with mlflow.start_run(run_name=f"train_{analysis_name}"):

        model_name = f"{MODEL_SPACE_PREFIX}_{analysis_name}"

        # 1) important parameters
        mlflow.log_params({
            "analysis": analysis_name,
            "seed": ml_parameters["seed"],
            "kmeans_k": ml_parameters["kmeans_k"],
            "pca_k": ml_parameters["pca_k"],
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        })

        # 2) (opcional) log list of features
        mlflow.log_text("\n".join(X.columns.tolist()), "features.txt")

        # 3) save the pipeline as a "model" in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model_name,
            input_example=X_example,
            signature=signature
        )

    return pipeline, X, ml_parameters, user_ids

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract PCA-transformed data and cluster IDs
# MAGIC ###### Apply the full transform chain manually to extract PCA components

# COMMAND ----------

def predict_X(pipeline: Pipeline, X, user_ids: pd.Series):
    feature_names = X.columns.tolist()

    X_imputed = pipeline["imputer_zero"].transform(X)
    X_scaled = pipeline["scaler"].transform(X_imputed)
    X_pca = pipeline["pca"].transform(X_scaled)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    valid_mask = ~np.isnan(X_scaled_df).any(axis=0)
    feature_names_final = X.columns[valid_mask]

    cluster_ids = pipeline["kmeans"].predict(X_pca)

    real_pca_k = pipeline["pca"].n_components_

    # Explained variance ratio (one value per PCA component)
    explained = pipeline["pca"].explained_variance_ratio_
    # Defensive: ensure length matches real_pca_k
    explained = explained[:real_pca_k]

    pca_components = pd.DataFrame(
        pipeline["pca"].components_,
        columns=feature_names_final
    ) # each row is one PCA component
    pca_components["pca_id"] = [f"pca_{i+1}" for i in range(real_pca_k)]
    pca_components["explained_variance_ratio"] = explained
    pca_components["explained_variance_ratio_cum"] = np.cumsum(explained)

    predict_df = pd.DataFrame(
        X_pca,
        columns=[f"pca_{i+1}" for i in range(real_pca_k)]
    )
    predict_df.insert(0, "unique_user_id", user_ids.values)
    predict_df["cluster_id"] = [f"cluster_{c}" for c in cluster_ids]
    

    return pca_components, predict_df

# analysis = 'analysis_2'
# pipeline, X = learn_X_into_pipeline(analysis)
# pca_components, predict_df = predict_X(pipeline, X)


# COMMAND ----------

def train_predict_analysis(analysis: str, ml_parameters={"kmeans_k":FIXED_KMEANS_K, "seed":RANDOM_SEED}):
    analysis_name = ANALYSIS_NAMES[analysis]
    
    pipeline, X, updated_params, user_ids = learn_X_into_pipeline(analysis, ml_parameters)
    pca_components, predict_df = predict_X(pipeline, X, user_ids)

    new_pca_k = updated_params["pca_k"]
    
    pca_spark = spark.createDataFrame(pca_components)
    predict_spark = spark.createDataFrame(predict_df)
    
    (
        pca_spark
        .withColumn("created_at", F.current_timestamp())
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}")
    )

    (
        predict_spark
        .withColumn("created_at", F.current_timestamp())
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")
    )

    return new_pca_k


# COMMAND ----------

def train_predict_all_analysis(ml_parameters={"kmeans_k":FIXED_KMEANS_K, "seed":RANDOM_SEED}):
    new_pca_k_dict = {}

    for analysis, analysis_name in ANALYSIS_NAMES.items():
        new_pca_k_dict[analysis] = train_predict_analysis(analysis, ml_parameters=ml_parameters)

    return new_pca_k_dict
    

# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC select      count(1)
# MAGIC from        uat_analytics.user_segmentation.feat_online_casino

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate all possibilities - all analysis
# MAGIC ###### (find out better parameters)

# COMMAND ----------


def evaluate_missing_k(analysis: str):

    if spark.catalog.tableExists(T_GOLD_CUSTOMER_EVAL_SCORE):
        processed_kmeans_ks = set(
            row["kmeans_k"] for row in spark.table(T_GOLD_CUSTOMER_EVAL_SCORE).filter(F.col("analysis_id") == analysis).select("kmeans_k").distinct().collect()
        )
    else:
        processed_kmeans_ks = set()

    missing_kmeans_ks = list(set(KMEANS_TRY_RANGE) - processed_kmeans_ks)

    # return missing_kmeans_ks
    
    if missing_kmeans_ks:
        evaluate_kmeans_k_list(analysis=analysis, kmeans_try_list=missing_kmeans_ks)

# evaluate_missing_k('analysis_5')
        

# COMMAND ----------

def evaluate_all_missing_k():
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        evaluate_missing_k(analysis)
        
evaluate_all_missing_k()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and predict with the best parameters

# COMMAND ----------

train_predict_all_analysis()

# COMMAND ----------

# MAGIC %md
# MAGIC ### All the tested Kmeans k's:

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_silhouette_by_k(analysis: str, agg: str = "max", limit_k: int | None = None, y_pad_ratio: float = 0.05):

    df = (
        spark.table(f"{T_GOLD_CUSTOMER_EVAL_SCORE}")
            .filter(F.col("analysis_id") == analysis)
            .select(
                F.col("kmeans_k").cast("int").alias("k_means_k"),
                F.col("sil_score").cast("double").alias("sil_score")
            )
            .filter(F.col("k_means_k").isNotNull() & F.col("sil_score").isNotNull())
    )

    if limit_k is not None:
        df = df.filter(F.col("k_means_k") <= F.lit(int(limit_k)))

    agg = agg.lower()
    agg_map = {
        "max": F.max("sil_score").alias("sil_score"),
        "avg": F.avg("sil_score").alias("sil_score"),
        "mean": F.avg("sil_score").alias("sil_score"),
        "min": F.min("sil_score").alias("sil_score"),
        "first": F.first("sil_score", ignorenulls=True).alias("sil_score"),
    }
    if agg not in agg_map:
        raise ValueError(f"agg inválido: {agg}. Use um de: {list(agg_map.keys())}")

    pdf = (
        df.groupBy("k_means_k")
          .agg(agg_map[agg])
          .orderBy("k_means_k")
          .toPandas()
    )

    if pdf.empty:
        raise ValueError("No data available do plot (after filtering).")

    # --- adjusting Y axis to data range ---
    y_min = float(pdf["sil_score"].min())
    y_max = float(pdf["sil_score"].max())
    y_range = y_max - y_min

    # proportional padding
    pad = max(y_range * float(y_pad_ratio), 1e-4) if y_pad_ratio is not None else 1e-4

    plt.figure(figsize=(10, 4))
    plt.bar(pdf["k_means_k"], pdf["sil_score"])
    plt.xlabel("k_means_k")
    plt.ylabel("sil_score")
    plt.title("Silhouette score por K (K-Means)")
    plt.xticks(pdf["k_means_k"].tolist())

    if y_pad_ratio == 0:
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(y_min - pad, y_max + pad)

    plt.tight_layout()
    plt.show()

    return pdf




# COMMAND ----------


for analysis, analysis_name in ANALYSIS_NAMES.items():
    plot_silhouette_by_k(analysis, agg="avg")
