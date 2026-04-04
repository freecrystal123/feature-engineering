# Databricks notebook source
# MAGIC %md
# MAGIC # User Segmentation
# MAGIC ## (Gold: fact_wager, fact_deposit, fact_withdrawal, dim_user, dim_coupon)
# MAGIC
# MAGIC Pipeline:
# MAGIC 1. Build metrics A–E from Gold tables:
# MAGIC    - A: Engagement → fact_wager
# MAGIC    - B: Financial → fact_wager, fact_deposit, fact_withdrawal
# MAGIC    - C: Risk/Performance → fact_wager
# MAGIC    - D: RFM → fact_wager
# MAGIC    - E: Context/Profile → dim_user, dim_coupon
# MAGIC 2. Normalize features
# MAGIC 3. PCA
# MAGIC 4. K-Means clustering
# MAGIC 5. Cluster summary + evaluation
# MAGIC 6. Save to Gold's Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ![diagram.png](./diagram.png "diagram.png")
# MAGIC
# MAGIC ## Everything highlighted in yellow indicates that there is a standardized code that handles all cases together, without needing to do them one by one.

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation 

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

write_tables_list = [T_GOLD_CUSTOMER_FEATURES_PREFIX,
                    T_GOLD_CUSTOMER_PCA_COMP_PREFIX,
                    T_GOLD_CUSTOMER_PREDICTED_PREFIX,
                    T_GOLD_CUSTOMER_EVAL_SCORE,
                    T_GOLD_CUSTOMER_CLUSTERS_PREFIX,
                    T_GOLD_CUSTOMER_SUGGESTIONS_PREFIX]

# COMMAND ----------

def drop_analysis(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    analysis_tables = (
            spark.sql(f"SHOW TABLES IN {WRITE_SCHEMA}")
                .filter(F.col("tableName").like(f"%{analysis_name}%"))
                .withColumn(
                    "table",
                    F.concat(F.col("database"), F.lit("."), F.col("tableName"))
                )
                .select("table")
                .collect()
            )

    for table in analysis_tables:
        spark.sql(f"DROP TABLE IF EXISTS {WRITE_CATALOG}.{table["table"]}")

    spark.sql(f"DELETE FROM {T_GOLD_CUSTOMER_EVAL_SCORE} WHERE analysis_id = '{analysis}'")


# COMMAND ----------


def reset_env():
    for analysis_id, analysis_names in ANALYSIS_NAMES.items():
        drop_analysis(analysis_id)

# COMMAND ----------

# MAGIC %skip
# MAGIC reset_env()

# COMMAND ----------

# MAGIC %skip
# MAGIC %run ./01_feature_engineering_online_casino

# COMMAND ----------

# MAGIC %run ./021_model_training

# COMMAND ----------

# MAGIC %run ./031_clusters_name

# COMMAND ----------

# MAGIC %skip
# MAGIC %run ./04_suggestions