# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 - suggestions (actions to do with each cluster)

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

# MAGIC %run ./00_llm_utils

# COMMAND ----------
# MAGIC %run ../utils

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, StructType, StructField
from delta.tables import DeltaTable
import json

import re

# COMMAND ----------

def get_pca_metadata(analysis: str) -> dict:
    analysis_name = ANALYSIS_NAMES[analysis]

    pca_table = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"

    pca_df = (
        read_current_table(spark=spark, table_base=pca_table)
        .select("pca_id", "pca_component_name", "pca_component_description")
    )

    # Map: pca_id -> {name, description}
    pca_meta_map = {
        row["pca_id"]: {
            "name": row["pca_component_name"],
            "description": row["pca_component_description"]
        }
        for row in pca_df.collect()
    }
    return pca_meta_map

# COMMAND ----------

suggestions_prompt = '''
You are a strategic advisor for an online gaming platform.

You are given a list of customer clusters.
Each cluster includes:
- Cluster name
- Cluster description
- Key behavioral characteristics (risk, volatility, activity, profitability, promotion usage)
- Number of users

Your task is to suggest appropriate, high-level actions
for MULTIPLE internal teams, based on each cluster's behavior.

You must provide SEPARATE recommendations for each team.
Do NOT mix responsibilities across teams.

Teams to address:
1. Marketing
2. CRM / Retention
3. Product
4. Responsible Gaming

General guidelines:
- Base all suggestions strictly on the cluster behavior provided.
- Keep actions realistic, ethical, and aligned with responsible gaming principles.
- Do NOT encourage excessive risk-taking.
- Avoid campaign copy or detailed execution plans.
- Use clear, simple language.

For EACH cluster, provide:

Marketing:
- Objective (1 sentence)
- Recommended actions (2–3 bullets)

CRM / Retention:
- Objective (1 sentence)
- Recommended actions (2–3 bullets)

Product:
- Objective (1 sentence)
- Recommended actions (2–3 bullets)

Responsible Gaming:
- Objective (1 sentence)
- Recommended actions (2–3 bullets)

Rules:
- If a cluster shows high or very high risk, Responsible Gaming actions are mandatory.
- If a cluster is an outlier or very small, clearly state that actions should be cautious.
- Do NOT invent user behavior beyond what is described.
- Do NOT suggest aggressive incentives for high-risk clusters.

MANDATORY OUTPUT FORMAT:
<<<RESULT>>>
cluster_id_1=<ID>

marketing_objective_1=<TEXT>
marketing_actions_1=
- <ACTION>
- <ACTION>

crm_objective_1=<TEXT>
crm_actions_1=
- <ACTION>
- <ACTION>

product_objective_1=<TEXT>
product_actions_1=
- <ACTION>
- <ACTION>

rg_objective_1=<TEXT>
rg_actions_1=
- <ACTION>
- <ACTION>

cluster_id_2=...
...
<<<END>>>
'''


# COMMAND ----------

def get_clusters(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    clusters_table = f'{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}'

    return read_current_table(spark=spark, table_base=clusters_table)

# get_clusters('analysis_1').display()


# COMMAND ----------

def is_strong(intensity: str) -> bool:
    return intensity in ("high", "very_high")


def build_cluster_action_input(df):
    lines = []
    pca_meta_map = get_pca_metadata("analysis_1")

    for row in df.collect():
        cluster_id = row["cluster_id"]
        num_users = row["num_users"]

        lines.append(f"Cluster ID: {cluster_id}")
        lines.append(f"Cluster Name: {row['cluster_name_refined']}")
        lines.append(f"Users: {num_users}")
        lines.append("")
        lines.append("Description:")
        lines.append(row["cluster_description_refined"])
        lines.append("")
        lines.append("Dominant characteristics:")

        has_any = False
        for pca_id, meta in pca_meta_map.items():
            intensity_col = f"{pca_id}_intensity"
            if intensity_col not in row:
                continue

            intensity = row[intensity_col]
            if is_strong(intensity):
                lines.append(f"- {meta['name']} intensity: {intensity}")
                has_any = True

        if not has_any:
            lines.append("- No dominant high-intensity characteristics")

        if num_users <= 10:
            lines.append("")
            lines.append(
                "Note: This cluster is an extreme outlier with very few users. "
                "Suggested actions should be cautious and non-scaled."
            )

        lines.append("\n---\n")

    return "\n".join(lines)

# COMMAND ----------

def get_suggestions(analysis: str):
    clusters_info = get_clusters('analysis_1')

    action_input = build_cluster_action_input(clusters_info)

    suggestions = get_llm_response('analysis_1',suggestions_prompt,action_input)

    variables = ['cluster_id_',
                'marketing_objective_',
                'marketing_actions_',
                'crm_objective_',
                'crm_actions_',
                'product_objective_',
                'product_actions_',
                'rg_objective_',
                'rg_actions_']
    suggestions_json = extract_variables_from_block(suggestions, variables, flag_several=True)
    suggestions_df = spark.createDataFrame(suggestions_json).drop('index')

    clusters = get_clusters(analysis)


    clusters = (
        clusters
            .select(
                F.col("cluster_id").alias("cluster_id"),
                F.col("cluster_name_refined").alias("cluster_name"),
                F.col("cluster_description_refined").alias("cluster_summary"),
                F.col("num_users").alias("users")
            )
            .join(
                suggestions_df,
                'cluster_id',
                'left'
            )
    )

    return clusters


def save_suggestions(suggestions_df, analysis):
    analysis_name = ANALYSIS_NAMES[analysis]
    publish_result = publish_table(
        spark=spark,
        df=suggestions_df.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_GOLD_CUSTOMER_SUGGESTIONS_PREFIX}_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="table",
    )
    print(
        f"Published suggestions: {publish_result['snapshot_table']} and "
        f"{publish_result['current_table']} (run_id={publish_result['run_id']})"
    )


# COMMAND ----------


def get_all_suggestions():
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        suggestions_df = get_suggestions(analysis)
        displayHTML(f"## {analysis_name}:")
        display(suggestions_df)
        save_suggestions(suggestions_df, analysis)

# COMMAND ----------

get_all_suggestions()
