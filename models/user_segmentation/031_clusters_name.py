# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 - clusters naming

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

# MAGIC %run ./00_llm_utils

# COMMAND ----------

# MAGIC %run ./01_feature_configs

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, StructType, StructField
from delta.tables import DeltaTable
import json
from databricks_genai_inference import ChatCompletion
import re

# COMMAND ----------

TOP_N_FEATURES = 6 # top variables for PCA components naming
TOP_N_PCA_COMP = 0 # all

NAME_PCAS = True
NAME_CLUSTERS = True

# COMMAND ----------

pca_component_naming_prompt = '''
You are given one or more PCA components.
Each PCA component is provided as a SINGLE JSON object, one per line.

The subject is <<SUBJECT>>

Each JSON object contains:
- pca_id
- stats:
  - top_1_share
  - top_3_share
  - num_features_above_threshold
- top_features: ranked features with PCA loadings

Each PCA component MUST be interpreted independently.
Do NOT mix information across components.

GOAL
Assign a clear, human-friendly name to EACH PCA component that describes the
dominant BEHAVIORAL DIMENSION (axis of variation) captured by its strongest loadings.

HOW TO INTERPRET PCA
- The SIGN of a loading (+/-) indicates direction only and may flip across runs.
- Focus on MAGNITUDE (absolute value), not sign.
- Stats are SUPPORTING signals only:
  - High top_1_share → single dominant driver
  - High top_3_share → concentrated dimension
  - Higher num_features_above_threshold → broader dimension
- Do NOT imply “more is better” or “more is worse”.

CRITICAL RULE: SIGN-INVARIANT NAMING
- PCA names MUST remain valid if the sign of the component flips.
- Avoid directional or interpretive terms such as:
  “recent”, “inactive”, “winning”, “losing”, “high”, “low”.
- Prefer neutral, scale-free dimensions, for example:
  - Time Since Activity (instead of Recency or Inactivity)
  - Engagement Decay
  - Outcome Variability
  - Activity Intensity

CRITICAL RULE: PCA ≠ SEGMENTS
- Do NOT name user groups or populations.
- PCA names must describe WHAT VARIES, not WHO the users are.

CRITICAL RULE: MONEY DIRECTION
Money-related terms MUST be unambiguous.

DEFINITIONS (MANDATORY):
- Win / Winning → money flows TO THE PLAYER
- Profit / Revenue / NGR → money flows TO THE OPERATOR

RULES
1) NEVER use "Profitable" alone.
2) If money direction CANNOT be determined:
   - DO NOT use win, profit, revenue, or loss terminology.
   - Use neutral dimensions such as:
     Outcome Variability, Outcome Dispersion, Risk, Volatility,
     Stake Management, Loss Tolerance.
3) Avoid judgmental or moral language.

TASK (for EACH PCA)
1. Analyze top_features by absolute loading magnitude.
2. Use stats only to assess concentration vs dispersion.
3. Identify ONE dominant behavioral dimension.
4. Assign a concise name (1–2 words).
5. Provide a short rationale (1–2 sentences).

NAMING RULES
- ONE concept per PCA.
- No conjunctions, slashes, or commas.
- Prefer standard behavioral analytics terms:
  Activity, Frequency, Time, Volatility, Risk,
  Promotions, Loss Tolerance, Stake Management.
- If names repeat, disambiguate with qualifiers:
  Frequency, Intensity, Variability, Magnitude, Stability, Momentum.

OUTPUT FORMAT (MANDATORY)
<<<RESULT>>>
PCA_ID_1=pca_1
PCA_NAME_1=<name>
RATIONALE_1=<1–2 sentences>

PCA_ID_2=pca_2
PCA_NAME_2=<name>
RATIONALE_2=<1–2 sentences>
...
<<<END>>>
'''

# COMMAND ----------

def get_feature_cols_from_features_table(analysis: str):

    analysis_name = ANALYSIS_NAMES[analysis]
    features_table = f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}"

    df = spark.table(features_table)

    exclude = {
        "unique_user_id",
        "created_at",
        # extras defensivos (não atrapalham se não existirem)
        "cluster_id",
        "pca_id",
        "rfm_segment",
    }

    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols
# display(get_feature_cols_from_features_table('analysis_1'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA analysing

# COMMAND ----------

def get_pca_component_names_with_llm(analysis:str, pca_info: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    attempt = 1
    while attempt <= LLM_ATTEMPTS:

        feature_descriptions = get_feature_descriptions(analysis_name)

        feature_descriptions_json = json.dumps(feature_descriptions, ensure_ascii=False, sort_keys=True)

        llm_pca_payload = f"""BEGIN_FEATURE_DESCRIPTIONS
                            {feature_descriptions_json}
                            END_FEATURE_DESCRIPTIONS

                            BEGIN_PCA_PAYLOAD
                            {pca_info.strip()}
                            END_PCA_PAYLOAD
                            """

        message = get_llm_response(analysis=analysis, prompt=pca_component_naming_prompt, user_query=llm_pca_payload)
        print(message)

        variables = extract_variables_from_block(message, ['PCA_ID_', 'PCA_NAME_', 'RATIONALE_'],True)
        print(variables)

        result = []
        for m in variables:
            result.append({'pca_id': m['PCA_ID'], 'pca_name': m['PCA_NAME'], 'rationale': m['RATIONALE']})

        if len(result) > 0:
            return result
        attempt += 1
    
    print(f'Was not possible to define the component names after {attempt - 1} attempts. Following LLM answer:')
    print(message)
    print('Extracted variables:')
    print(variables)

    return []

# features = '["{\"rank_in_pca\":1,\"feature_name\":\"active_weeks_3m\",\"weight\":0.363225530973928}","{\"rank_in_pca\":2,\"feature_name\":\"bet_days_90d\",\"weight\":0.36234608324577644}","{\"rank_in_pca\":3,\"feature_name\":\"F_norm\",\"weight\":0.3592722448759041}","{\"rank_in_pca\":4,\"feature_name\":\"num_active_days\",\"weight\":0.33008255398698955}","{\"rank_in_pca\":5,\"feature_name\":\"rfm_score\",\"weight\":0.3290494871397899}","{\"rank_in_pca\":6,\"feature_name\":\"R_norm\",\"weight\":0.28373715976205205}","{\"rank_in_pca\":7,\"feature_name\":\"deposit_days_count\",\"weight\":0.26235188396804693}","{\"rank_in_pca\":8,\"feature_name\":\"num_bets_total\",\"weight\":0.2609798604909492}"]'

# a = get_pca_component_names_with_llm(analysis='analysis_1',features=features)

# print(a)

# COMMAND ----------

def get_pca_basic_info(analysis:str):
    analysis_name = ANALYSIS_NAMES[analysis]
    pca_table = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"

    pca_df = spark.table(pca_table).drop("pca_component_name", "pca_component_description")

    return pca_df

# print('PCA basic info:')
# get_pca_basic_info('analysis_1').display()

# COMMAND ----------


def get_pca_weights_unpivoted(df, analysis):
    # allowlist: only columns that exist in the features table
    feature_cols = get_feature_cols_from_features_table(analysis)

    # safety: keep only columns that actually exist in the PCA table
    feature_cols = [c for c in feature_cols if c in df.columns]

    expr = ", ".join([f"'{c}', {c}" for c in feature_cols])

    return (
        df.select(
            "pca_id",
            F.expr(f"stack({len(feature_cols)}, {expr}) as (feature_name, weight)")
        )
    )
    # print('get_pca_weights')
    # pca_long.display()
    # return pca_long
# get_pca_basic_info('analysis_1').transform(get_pca_weights_unpivoted,'analysis_1').display()

# COMMAND ----------


def get_pca_ranked(df):
    
    pca_ranked = (
        df
        .withColumn("abs_weight", F.abs("weight"))
        .withColumn(
            "rank_in_pca",
            F.row_number().over(
                Window.partitionBy("pca_id").orderBy(F.desc("abs_weight"))
            )
        )
    )
    return pca_ranked

# COMMAND ----------

def get_top_features(df):
    # pca_ranked = get_pca_ranked(analysis)
    top_features = (
        df
        .filter(F.col("rank_in_pca") <= TOP_N_FEATURES)
    )
    return top_features

# COMMAND ----------

# MAGIC %skip
# MAGIC def extract_general_pca_info(analysis: str):
# MAGIC     pca_basic_info = get_pca_basic_info(analysis)
# MAGIC     pca_weights = get_pca_weights_unpivoted(pca_basic_info)
# MAGIC
# MAGIC     return pca_weights
# MAGIC extract_general_pca_info('analysis_1').display()

# COMMAND ----------

def get_top_shares(df):

    # Exclude metadata rows that are not PCA loadings
    cleaned = (
        df
        .filter(~F.col("feature_name").isin("explained_variance_ratio", "explained_variance_ratio_cum"))
    )

    # Denominator: sum abs weights within TOP_N_FEATURES (after cleaning)
    shares = (
        cleaned
        .groupBy("pca_id")
        .agg(
            F.sum("abs_weight").alias("sum_abs_topn"),
            F.sum(F.when(F.col("rank_in_pca") == 1, F.col("abs_weight")).otherwise(F.lit(0.0))).alias("top1_abs"),
            F.sum(F.when(F.col("rank_in_pca") <= 3, F.col("abs_weight")).otherwise(F.lit(0.0))).alias("top3_abs"),
        )
        .withColumn("top_1_share", F.when(F.col("sum_abs_topn") > 0, F.col("top1_abs") / F.col("sum_abs_topn")))
        .withColumn("top_3_share", F.when(F.col("sum_abs_topn") > 0, F.col("top3_abs") / F.col("sum_abs_topn")))
        .select("pca_id", "top_1_share", "top_3_share")
        .orderBy("pca_id")
    )

    shares = shares.join(df, "pca_id")

    return shares

# COMMAND ----------

def get_num_features_above_threshold(df, threshold: float = 0.3):

    # Exclude non-loading rows if they exist in the unpivoted DF
    cleaned = df.filter(
        ~F.col("feature_name").isin(
            "explained_variance_ratio",
            "explained_variance_ratio_cum",
            "created_at",
            "pca_component_name",
            "pca_component_description"
        )
    )

    # Compute abs_weight and count how many are above threshold
    counts = (
        cleaned
        .withColumn("abs_weight", F.abs(F.col("weight")))
        .groupBy("pca_id")
        .agg(
            F.sum(F.when(F.col("abs_weight") >= F.lit(threshold), F.lit(1)).otherwise(F.lit(0)))
             .cast("int")
             .alias("num_features_above_threshold")
        )
        .orderBy("pca_id")
    )
    counts = counts.join(df,'pca_id')

    return counts

# (
# get_pca_basic_info('analysis_1')
# .transform(get_pca_weights_unpivoted)
# .transform(get_num_features_above_threshold)
# .display()
# )

# COMMAND ----------

def extract_all_pca_infos_to_llm(df, analysis: str):
    
    top_features_with_rank = (
        df
            .transform(get_pca_weights_unpivoted, analysis)
            .transform(get_num_features_above_threshold)
            .transform(get_pca_ranked)
            .transform(get_top_features)
            .transform(get_top_shares)

    )

    return top_features_with_rank

# print('All PCA infos:')
# get_pca_basic_info('analysis_1').transform(extract_all_pca_infos_to_llm, 'analysis_1').display()


# COMMAND ----------

def format_pca_payload_for_llm(df_pca_info) -> str:

    # Build a per-row json for top features
    df_pca_info = df_pca_info.select(
        "pca_id",
        "top_1_share",
        "top_3_share",
        "num_features_above_threshold",
        "rank_in_pca",
        "feature_name",
        "weight"
    )

    # Collect per PCA (small volume: num_pcas * TOP_N_FEATURES)
    rows = (
        df_pca_info
        .orderBy("pca_id", "rank_in_pca")
        .collect()
    )

    # Group in python to keep formatting exactly how we want
    by_pca = {}
    for r in rows:
        pca_id = r["pca_id"]

        if pca_id not in by_pca:
            by_pca[pca_id] = {
                "stats": {
                    "top_1_share": float(r["top_1_share"]) if r["top_1_share"] is not None else None,
                    "top_3_share": float(r["top_3_share"]) if r["top_3_share"] is not None else None,
                    "num_features_above_threshold": int(r["num_features_above_threshold"]) if r["num_features_above_threshold"] is not None else None,
                },
                "top_features": []
            }

        by_pca[pca_id]["top_features"].append({
            "rank_in_pca": int(r["rank_in_pca"]),
            "feature_name": r["feature_name"],
            "weight": float(r["weight"]) if r["weight"] is not None else None
        })

    # Serialize as a single prompt string (easy to parse and stable)
    blocks = []
    for pca_id in sorted(by_pca.keys()):
        payload = {
            "pca_id": pca_id,
            "stats": by_pca[pca_id]["stats"],
            "top_features": by_pca[pca_id]["top_features"],
        }
        blocks.append(json.dumps(payload, ensure_ascii=False))

    # One JSON per line
    return "\n".join(blocks)

# COMMAND ----------

# MAGIC %skip
# MAGIC df_all_info = get_pca_basic_info('analysis_1').transform(extract_all_pca_infos_to_llm,'analysis_1')
# MAGIC print(format_pca_payload_for_llm(df_all_info))

# COMMAND ----------

# MAGIC %skip
# MAGIC def format_top_features(df):
# MAGIC     formatted_top_features = (
# MAGIC         df
# MAGIC         .groupBy("pca_id")
# MAGIC         .agg(
# MAGIC             F.collect_list(
# MAGIC                 F.struct("rank_in_pca", "feature_name", "weight")
# MAGIC             ).alias("top_features")
# MAGIC         )
# MAGIC         .orderBy("pca_id")
# MAGIC     )
# MAGIC     
# MAGIC     return formatted_top_features

# COMMAND ----------


def get_pca_component_names_with_llm_from_analysis(df, analysis):
    
    all_pca_info = format_pca_payload_for_llm(df)

    components = get_pca_component_names_with_llm(analysis=analysis,pca_info=all_pca_info)
    # print(components)

    schema = StructType([
        StructField("pca_id", StringType(), False),
        StructField("pca_component_name", StringType(), True),
        StructField("pca_component_description", StringType(), True),
    ])

    components_df = spark.createDataFrame(
        components,
        schema=["pca_id", "pca_component_name", "pca_component_description"]
    )
    # components_df.display()
    return components_df



# COMMAND ----------

def enrich_pca_table_with_names(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    pca_table = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"

    # Get PCA component names and descriptions
    # pca_weights = get_pca_weights(analysis=analysis)
    pca_basic_info = get_pca_basic_info(analysis)

    names_df = (
        pca_basic_info
            .transform(extract_all_pca_infos_to_llm, analysis)
            .transform(get_pca_component_names_with_llm_from_analysis, analysis)
    )


    # Read PCA components table
    pca_df = spark.table(pca_table).drop(
        "pca_component_name",
        "pca_component_description"
    )

    # Join on pca_id to add name and description
    enriched_df = (
        pca_df.join(
            names_df,
            on="pca_id",
            how="left"
        )
    )


    # print('pca_df')
    # display(pca_df)

    print('PCA info with names:')
    display(enriched_df)



    delta_table = DeltaTable.forName(spark, pca_table)

    (
        delta_table.alias("t")
        .merge(
            enriched_df.alias("s"),
            "t.pca_id = s.pca_id"
        )
        .withSchemaEvolution()
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


# COMMAND ----------

def name_all_pca_analysis():
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        enrich_pca_table_with_names(analysis)

# COMMAND ----------

if NAME_PCAS:
    name_all_pca_analysis()

# COMMAND ----------

# MAGIC %skip
# MAGIC # df = spark.sql("""
# MAGIC # SELECT * EXCEPT(pca_component_name, created_at, pca_component_description)
# MAGIC # FROM uat_gold.machine_learning.pcacomp_online_casino
# MAGIC # """)
# MAGIC
# MAGIC pca_weights = get_pca_weights(analysis='analysis_1')
# MAGIC (
# MAGIC     pca_weights
# MAGIC     .transform(get_pca_ranked)
# MAGIC     .transform(get_top_features)
# MAGIC     .display()
# MAGIC )    
# MAGIC # print(all_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster Analysis

# COMMAND ----------

MIN_CLUSTER_SIZE = 50

# COMMAND ----------

def get_pca_metadata(analysis: str) -> dict:
    analysis_name = ANALYSIS_NAMES[analysis]

    pca_table = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"

    pca_df = (
        spark.table(pca_table)
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

# display(get_pca_metadata('analysis_1'))


# COMMAND ----------

def get_cluster_pca_stats(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    pred_df = spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")
    pca_columns = [c for c in pred_df.columns if c.startswith("pca_")]

    cluster_pca_stats = (
        pred_df
        .groupBy("cluster_id")
        .agg(
            *[F.avg(c).alias(f"{c}_mean") for c in pca_columns],
            *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_median") for c in pca_columns],
            F.count("*").alias("num_users")
        )
    )
    valid_clusters = cluster_pca_stats.filter(F.col("num_users") >= MIN_CLUSTER_SIZE)

    # --- GLOBAL population thresholds (across ALL users), not across clusters ---
    percentile_exprs = []
    for c in pca_columns:
        percentile_exprs.extend([
            F.expr(f"percentile_approx({c}, 0.10)").alias(f"{c}_p10"),
            F.expr(f"percentile_approx({c}, 0.30)").alias(f"{c}_p30"),
            F.expr(f"percentile_approx({c}, 0.70)").alias(f"{c}_p70"),
            F.expr(f"percentile_approx({c}, 0.90)").alias(f"{c}_p90"),
        ])

    global_pca_percentiles = pred_df.agg(*percentile_exprs)

    # attach those global thresholds to each cluster row
    cluster_pca_stats = cluster_pca_stats.crossJoin(global_pca_percentiles)

    return cluster_pca_stats
# get_pca_stats('analysis_1').display()

# COMMAND ----------


def calculate_pca_intensity_column(pca_column):
    mean = F.col(f"{pca_column}_mean")
    median = F.col(f"{pca_column}_median")

    p10 = F.col(f"{pca_column}_p10")
    p30 = F.col(f"{pca_column}_p30")
    p70 = F.col(f"{pca_column}_p70")
    p90 = F.col(f"{pca_column}_p90")

    def classify(x):
        return (
            F.when(x <= p10, F.lit("very_low"))
             .when(x <= p30, F.lit("low"))
             .when(x <= p70, F.lit("medium"))
             .when(x <= p90, F.lit("high"))
             .otherwise(F.lit("very_high"))
        )

    mean_label = classify(mean)
    median_label = classify(median)

    scale = F.create_map(
        F.lit("very_low"), F.lit(1),
        F.lit("low"), F.lit(2),
        F.lit("medium"), F.lit(3),
        F.lit("high"), F.lit(4),
        F.lit("very_high"), F.lit(5)
    )

    mean_score = scale.getItem(mean_label)
    median_score = scale.getItem(median_label)

    return (
        F.when(F.abs(mean_score - median_score) > 1, F.lit("neutral"))
         .otherwise(mean_label)
         .alias(f"{pca_column}_intensity")
    )


def add_intensity_columns(df):
    pca_columns = sorted({
        c.replace('_mean','')
        for c in df.columns
        if c.startswith("pca_")
        if c.endswith("_mean")
    })
    for c in pca_columns:
        df = df.withColumn(
            f"{c}_intensity",
            calculate_pca_intensity_column(c)
        )
    return df


# COMMAND ----------

cluster_naming_prompt = """
You are given the behavioral characteristics of a single customer cluster.
The subject is <<SUBJECT>>

The characteristics describe the intensity of multiple behavioral dimensions (PCA components).
Some dimensions are marked as "key characteristic", others as "neutral" or "weak presence".

You are also given:
- cluster size (num_users)
- mean/median signals for the key dimensions

Your task is to:
1. Identify the dominant behavioral pattern of this cluster.
2. Assign a concise and intuitive cluster name (maximum 1–3 words).
3. Write a short cluster description (1–2 sentences) explaining the main behavior.

Strict rules:
- Focus primarily on dimensions marked as "key characteristic".
- Use "neutral" dimensions only for context.
- Ignore dimensions marked as "weak presence" unless they strongly contrast with a key one.
- Do NOT list PCA names in the output.
- Do NOT mention intensity labels (high, low, etc.) explicitly.
- Avoid generic names (e.g., "Mixed Users", "Average Players").
- Prefer clear behavioral terms commonly used in analytics.

Important naming rule (to avoid duplicates):
- If the cluster is an OUTLIER compared to typical clusters (e.g., unusually extreme mean/median signals and/or very small size),
  you MUST reflect that using one strong qualifier such as: "Extreme", "Ultra", or "Whale".
- If two clusters share the same dominant pattern, differentiate them by adding one qualifier (e.g., "Frequent" vs "Ultra", "Active" vs "Extreme").

Output format (mandatory):
<<<RESULT>>>
CLUSTER_NAME=<NAME1>
CLUSTER_DESCRIPTION=<DESCRIPTION1>
<<<END>>>
"""

# COMMAND ----------

def get_cluster_names_with_llm(analysis: str, cluster_characteristics: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    attempt = 1
    while attempt <= LLM_ATTEMPTS:
        try:
            message = get_llm_response(analysis=analysis, prompt=cluster_naming_prompt, user_query=cluster_characteristics)

            variables = extract_variables_from_block(message, ['CLUSTER_NAME', 'CLUSTER_DESCRIPTION'], False)

            cluster_name = variables[0]['CLUSTER_NAME']
            cluster_description = variables[0]['CLUSTER_DESCRIPTION']

            return {
                "cluster_name": cluster_name,
                "cluster_description": cluster_description
            }
        except:
            attempt += 1

    print("Was not possible to define the cluster name. Following LLM answer:")
    print(message)
    return None


# pca_name, rationale = get_pca_component_names_with_llm(analysis='analysis_1',features=features)

# cluster_characteristics = '''
# The cluster characteristics are:
#   - Activity has a high intensity (key characteristic). It refers to Features relate to user engagement and activity..
#   - Profitability has a low intensity (weak presence). It refers to Dominated by win/loss rates and net profit ratio..
#   - Risk_Timing has a high intensity (key characteristic). It refers to Combines risk and recency features..
#   - Risk_Behavior has a very_low intensity (weak presence). It refers to Focused on risk and betting behavior..
#   - Volatility has a low intensity (weak presence). It refers to Features indicate variability in user behavior..
#   - Timing has a medium intensity (neutral). It refers to Dominant features relate to time between bets and recency..
#   - Volatility has a very_low intensity (weak presence). It refers to Dominated by stake variability..
#   - Stake_Ratio has a neutral intensity (neutral). It refers to Highly dominated by stake to deposit ratio..
#   - Promotion_Usage has a medium intensity (neutral). It refers to Dominant feature is the number of coupons used..
#   '''

# print(get_cluster_names_with_llm('analysis_1',cluster_characteristics))

# COMMAND ----------

def intensity_role(intensity: str) -> str:
    if intensity in ("very_high", "high"):
        return "key characteristic"
    if intensity in ("very_low", "low"):
        return "weak presence"
    return "neutral"


def get_cluster_characteristics_str(cluster, pca_name_map, pca_cols):
    cluster_characteristics = ["The cluster characteristics are:"]

    for pca_id in pca_cols:
        pca_meta = pca_name_map.get(pca_id)
        if not pca_meta:
            continue

        pca_name = pca_meta["name"]
        pca_description = pca_meta["description"]

        intensity_col = f"{pca_id}_intensity"
        intensity = cluster[intensity_col] if intensity_col in cluster else "N/A"
        role = intensity_role(intensity)

        cluster_characteristics.append(
            f"  - {pca_name} has a {intensity} intensity ({role}). "
            f"It refers to {pca_description}."
        )

    cluster_characteristics_str = "\n".join(cluster_characteristics)

    return cluster_characteristics_str



# COMMAND ----------


def intensity_score(intensity: str) -> int:
    scale = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
    return scale.get(intensity, 3)

def get_cluster_characteristics_str(cluster, pca_name_map, pca_cols, top_n=4):
    items = []

    if top_n == 0:
        top_n = len(pca_cols)

    for pca_id in pca_cols:
        pca_meta = pca_name_map.get(pca_id)
        if not pca_meta:
            continue

        intensity_col = f"{pca_id}_intensity"
        intensity = cluster[intensity_col] if intensity_col in cluster else "medium"

        # importance = distance from "medium"
        score = abs(intensity_score(intensity) - 3)

        # Keeps only really informative PCs
        if intensity in ("high", "very_high", "low", "very_low"):
            items.append((score, pca_id, intensity, pca_meta))

    # Strongest first
    items.sort(key=lambda x: x[0], reverse=True)
    items = items[:top_n]

    # cluster_characteristics = ["The cluster characteristics are:"]
    cluster_characteristics = [f"Cluster size: {cluster['num_users']} users.", "The cluster characteristics are:"]
    for _, pca_id, intensity, pca_meta in items:
        role = intensity_role(intensity)
        mean = cluster[f"{pca_id}_mean"]
        median = cluster[f"{pca_id}_median"]
        cluster_characteristics.append(
            f"  - {pca_meta['name']} has a {intensity} intensity ({role}). "
            f"mean={mean:.3f}, median={median:.3f}. "
            f"It refers to {pca_meta['description']}."
        )


    # fallback: if nothing passed, accept 2 any PCs as neutral
    if len(items) == 0:
        cluster_characteristics.append("  - No strong PCA signals detected (mostly neutral intensities).")

    return "\n".join(cluster_characteristics)



# COMMAND ----------

def add_cluster_names(df, analysis: str):

    pca_name_map = get_pca_metadata(analysis)

    # Identify PCA columns (remove duplicates _intensity)
    pca_cols = sorted({
        c.replace("_intensity", "")
        for c in df.columns
        if c.endswith("_intensity") and c.startswith("pca_")
    })

    cluster_name_rows = []
    # Prompt building (cluster by cluster)
    for cluster in df.collect():
        cluster_id = cluster["cluster_id"]

        cluster_characteristics_str = get_cluster_characteristics_str(cluster, pca_name_map, pca_cols, top_n=TOP_N_PCA_COMP)
        # print(cluster_characteristics_str)
        
        cluster_names_dict = get_cluster_names_with_llm(analysis=analysis,
                                                        cluster_characteristics=cluster_characteristics_str)
        
        cluster_name_rows.append(
            (
                cluster_id,
                cluster_names_dict["cluster_name"],
                cluster_names_dict["cluster_description"]
            )
        )
        

    clusters_names_df = spark.createDataFrame(
        cluster_name_rows,
        ["cluster_id", "cluster_name", "cluster_description"]
    )
    cluster_stats = df.join(clusters_names_df,"cluster_id","left")

    return cluster_stats




# COMMAND ----------

refining_cluster_naming_prompt = '''
You are given a list of customer clusters.
Some clusters may currently share similar names.

Each cluster includes:
- Behavioral information derived from PCA components
  (e.g. activity, risk, risk persistence, profitability, volatility, recency, promotion usage)
- The key original variables that explain each dominant PCA component
  (e.g. risk_score, max_loss_streak, stake_to_deposit_ratio, interval_volatility).

Your task is to:
1. Review ALL clusters together to understand their similarities and differences.
2. Assign a UNIQUE name to EACH cluster based on the dominant PCA components.
3. If multiple clusters share the same base behavioral concept,
   you MUST explicitly differentiate them using PCA-driven characteristics such as:
   - intensity (e.g. high vs medium)
   - persistence vs episodic behavior
   - volatility vs stability
   - timing or recency patterns
   - focus on profitability, promotions, or stakes

IMPORTANT DIFFERENTIATION RULE:
- If two or more clusters are behaviorally similar,
  you MUST explain the difference using the PCA components AND the variables
  that drive those components.
- Do NOT use synonyms alone to differentiate clusters.
- If no clear behavioral difference exists, keep the same base name
  and add a clear PCA-based qualifier (e.g. Persistent, Volatile, Promotion-Focused).

LANGUAGE AND EXPLANATION RULES (MANDATORY):
- Do NOT reference internal PCA labels or variable group names
  (e.g. Risk_Behavior, Timing, Promotion_Usage).
- Always explain behavior in plain language derived from the variables
  (e.g. "long loss streaks", "irregular betting intervals", "high stakes relative to deposits").
- Reference at most 2–3 variables per cluster description.
- Use variables as causal evidence, not as a raw list.

OUTLIER RULE:
- If a cluster has very few users or represents an extreme case,
  clearly state that it reflects an outlier or extreme behavior.
- Keep the description short and avoid over-detailed explanations for such clusters.

Rules:
- Do NOT rename clusters unnecessarily.
- Keep names concise (1–3 words).
- Names must reflect actual behavioral differences, not wording variety.
- No two clusters may have the same name.
- Cluster descriptions must be short (1–2 sentences) and behavior-focused.
- Do NOT invent behaviors or variables that are not supported by the data.
- Use clear, simple language suitable for non-technical stakeholders.

MANDATORY OUTPUT FORMAT:
- The output MUST start with <<<RESULT>>> and end with <<<END>>>.
- For the clusters, return:
  cluster_id_1=<ID>
  cluster_name_1=<NAME>
  cluster_description_1=<DESCRIPTION>
  cluster_id_2=<ID>
  cluster_name_2=<NAME>
  cluster_description_2=<DESCRIPTION>
- Return ALL clusters, one after another.
- Do NOT include any text outside the defined format.
'''

# COMMAND ----------

def is_strong(intensity: str) -> bool:
    return intensity in ("high", "very_high")

def build_cluster_naming_input(df, analysis):
    lines = []
    pca_meta_map = get_pca_metadata(analysis)

    for row in df.collect():
        cluster_id = row["cluster_id"]

        lines.append(f"Cluster {cluster_id}:")
        lines.append(f"Current name: {row['cluster_name']}")
        lines.append("Key characteristics:")

        for pca_id, meta in pca_meta_map.items():
            intensity_col = f"{pca_id}_intensity"
            if intensity_col not in row:
                continue

            intensity = row[intensity_col]
            if is_strong(intensity):
                lines.append(f"- {meta['name']}: {intensity}")

        lines.append(f"Users: {row['num_users']}")
        lines.append(f"Notes: {row['cluster_description']}")
        lines.append("")

    return "\n".join(lines)



# COMMAND ----------

def add_refined_cluster_names(df, analysis):
    cluster_naming_input = build_cluster_naming_input(df, analysis)
    refined_cluster_names = get_llm_response(analysis=analysis,prompt=refining_cluster_naming_prompt, user_query=cluster_naming_input)

    variables = extract_variables_from_block(refined_cluster_names, ['cluster_id_', 'cluster_name_', 'cluster_description_'],True)

    refined_clusters_df = spark.createDataFrame(
        [
            (
                v["cluster_id"],
                v["cluster_name"],
                v["cluster_description"]
            )
            for v in variables
        ],
        ["cluster_id", "cluster_name_refined", "cluster_description_refined"]
    )
    final_cluster_df = df.join(refined_clusters_df, "cluster_id","left")
    return final_cluster_df
    

# COMMAND ----------

def get_cluster_names(analysis: str):
    cluster_pca_stats = get_cluster_pca_stats(analysis)

    cluster_names = (
        cluster_pca_stats
            .transform(add_intensity_columns)
            .transform(add_cluster_names, analysis=analysis)
            .transform(add_refined_cluster_names, analysis=analysis)
    )
    return cluster_names

# COMMAND ----------

def save_cluster_names(cluster_names, analysis):
    analysis_name = ANALYSIS_NAMES[analysis]
    (
        cluster_names
        .withColumn("created_at", F.current_timestamp())
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}")
    )

# COMMAND ----------

def name_cluster_analysis(analysis: str):
    cluster_names = get_cluster_names(analysis)
    save_cluster_names(cluster_names, analysis)

# COMMAND ----------

def name_all_cluster_analysis():
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        name_cluster_analysis(analysis)


# COMMAND ----------

if NAME_CLUSTERS:
    name_all_cluster_analysis()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clusters visualization (3 PCA components)

# COMMAND ----------

# %skip
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_clusters(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    pred_df = spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")

    pdf = (
        pred_df
        .select("cluster_id", "pca_1", "pca_2", "pca_3")
        
        .sample(withReplacement=False, fraction=0.02, seed=42)
        .toPandas()
    )

    cluster_colors = pd.Categorical(pdf["cluster_id"]).codes

    fig = plt.figure(figsize=(14, 6))

    pca_metadata = get_pca_metadata(analysis)
    pca_1_name = pca_metadata["pca_1"]['name']
    pca_2_name = pca_metadata["pca_2"]['name']
    pca_3_name = pca_metadata["pca_3"]['name']

    # --- LEFT: 2D PCA_1 vs PCA_2 ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(
        pdf["pca_1"],
        pdf["pca_2"],
        c=cluster_colors
    )
    ax1.set_xlabel(pca_1_name)
    ax1.set_ylabel(pca_2_name)
    ax1.set_title(f"{pca_1_name} vs {pca_2_name} (by cluster)")

    # --- RIGHT: 3D PCA_1 vs PCA_2 vs PCA_3 ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        pdf["pca_1"],
        pdf["pca_2"],
        pdf["pca_3"],
        c=cluster_colors
    )

    ax2.set_xlabel(pca_1_name)
    ax2.set_ylabel(pca_2_name)
    ax2.set_zlabel(pca_3_name)
    ax2.set_title(f"{pca_1_name} vs {pca_2_name} vs {pca_3_name} (by cluster)")

    plt.tight_layout()
    plt.show()

# COMMAND ----------

def plot_all_analysis_clusters():
    for analysis, analysis_name in ANALYSIS_NAMES.items():
        print(analysis_name.upper())
        plot_clusters(analysis)

# COMMAND ----------

plot_all_analysis_clusters()

# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC select      cluster_id, num_users, cluster_name, cluster_description, cluster_name_refined, cluster_description_refined
# MAGIC from        uat_gold.machine_learning.clusters_online_casino
# MAGIC order by    2 desc
# MAGIC

# COMMAND ----------



# COMMAND ----------

def get_cluster_metadata(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    clusters = spark.table(f'{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}')

    return {row['cluster_id']: row.asDict() for row in clusters.collect()}

# print(get_cluster_metadata('analysis_1'))


# COMMAND ----------



# COMMAND ----------

# MAGIC %skip
# MAGIC def get_all_cluster_radial_profiles():
# MAGIC     for analysis, analysis_name in ANALYSIS_NAMES.items():
# MAGIC         get_cluster_radial_profiles(analysis)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


def get_info_to_decide_merge_and_explain_clusters(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    features_df = spark.table(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}")
    pred_df     = spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}") 

    # 1) Join features + cluster_id
    base_df = (
        features_df.alias("f")
        .join(
            pred_df.select("unique_user_id", "cluster_id").alias("p"),
            on="unique_user_id",
            how="inner"
        )
    )

    # 2) selects only numeric
    meta_cols = {"unique_user_id", "cluster_id", "created_at", "updated_at", "analysis_id", "analysis_name"}
    numeric_feature_cols = [
        field.name for field in base_df.schema.fields
        if isinstance(field.dataType, NumericType) and field.name not in meta_cols
    ]

    # 3) Imputer zero (null -> 0)
    df_imp = base_df
    imp_cols = []
    for c in numeric_feature_cols:
        imp_name = f"imp_{c}"
        df_imp = df_imp.withColumn(imp_name, F.coalesce(F.col(c).cast("double"), F.lit(0.0)))
        imp_cols.append(imp_name)

    # 4) Scaler Fit
    agg_exprs = []
    for c in imp_cols:
        agg_exprs.append(F.avg(c).alias(f"mean__{c}"))
        agg_exprs.append(F.stddev_pop(c).alias(f"std__{c}"))
    stats = df_imp.agg(*agg_exprs).collect()[0].asDict()

    # 5) scaled columns
    df_scaled = df_imp
    scaled_cols = []
    for c in imp_cols:
        mean = float(stats.get(f"mean__{c}") or 0.0)
        std  = float(stats.get(f"std__{c}")  or 0.0)

        orig = c.replace("imp_", "")
        out = f"scaled_{orig}"
        scaled_cols.append(out)

        df_scaled = df_scaled.withColumn(
            out,
            F.when(F.lit(std) == 0.0, F.lit(0.0)).otherwise((F.col(c) - F.lit(mean)) / F.lit(std))
        )

    # =========================
    # A) Cluster profile (scaled) — better to decide merge
    # =========================
    profile_scaled_df = (
        df_scaled
        .groupBy("cluster_id")
        .agg(
            F.count("*").alias("num_users"),
            *[F.avg(c).alias(f"avg_{c}") for c in scaled_cols],
            *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"p50_{c}") for c in scaled_cols],
            *[F.stddev_pop(c).alias(f"std_{c}") for c in scaled_cols],
        )
        .orderBy(F.desc("num_users"))
    )

    displayHTML("## To decide merge:")
    display(profile_scaled_df)

    # =========================
    # B) Cluster profile (raw + lift) — better to explain better whats each cluster
    # =========================
    global_means = (
        df_imp.agg(*[F.avg(f"imp_{c}").alias(c) for c in numeric_feature_cols])
        .collect()[0]
        .asDict()
    )

    profile_raw_df = (
        df_imp
        .groupBy("cluster_id")
        .agg(
            F.count("*").alias("num_users"),
            *[F.avg(f"imp_{c}").alias(f"avg_{c}") for c in numeric_feature_cols],
            *[F.expr(f"percentile_approx(imp_{c}, 0.5)").alias(f"p50_{c}") for c in numeric_feature_cols],
        )
    )

    for c in numeric_feature_cols:
        profile_raw_df = profile_raw_df.withColumn(
            f"lift_{c}",
            F.col(f"avg_{c}") - F.lit(float(global_means.get(c) or 0.0))
        )

    displayHTML("## To explain the clusters:")
    display(profile_raw_df.orderBy(F.desc("num_users")))

# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC select      sum(vat_on_wager_amount) / sum(wager_amount)
# MAGIC from        prod_gold.finance.fact_wager fw
# MAGIC where       fw.source_system = "LOTTERY"

# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC select      sum(vat_on_deposit_amount) / sum(deposit_amount_usd)
# MAGIC from        prod_gold.finance.fact_deposit
# MAGIC
# MAGIC -- 0.042492555384507835 for IGaming
# MAGIC -- 0.048263809638196571 for Lottery
# MAGIC

# COMMAND ----------

# MAGIC %skip
# MAGIC from pyspark.sql import functions as F
# MAGIC import plotly.graph_objects as go
# MAGIC
# MAGIC # Exemplo (troque pelo seu DF real)
# MAGIC data = [
# MAGIC     ("cluster_2", "Dormant Users", "India", 599),
# MAGIC     ("cluster_2", "Dormant Users", "Philippines", 681),
# MAGIC     ("cluster_1", "Low Frequency Bettors", "Philippines", 698),
# MAGIC     ("cluster_2", "Dormant Users", "Nepal", 738),
# MAGIC     ("cluster_1", "Low Frequency Bettors", "India", 923),
# MAGIC     ("cluster_1", "Low Frequency Bettors", "Nepal", 1149),
# MAGIC ]
# MAGIC df = spark.createDataFrame(data, ["cluster_id", "cluster_name", "country", "user_count"])
# MAGIC
# MAGIC # (Opcional) garante tipos/limpeza
# MAGIC df = (
# MAGIC     df.select(
# MAGIC         "cluster_id",
# MAGIC         "cluster_name",
# MAGIC         F.trim(F.col("country")).alias("country"),
# MAGIC         F.col("user_count").cast("long").alias("user_count"),
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC df = spark.sql('''
# MAGIC select          c.cluster_id,
# MAGIC                 c.cluster_name_refined as cluster_name,
# MAGIC                 cri.country,
# MAGIC                 -- sum(case when up.nation_id is not null and cri.country is null then 1 else 0 end),
# MAGIC                 count(1) as user_count
# MAGIC from            uat_analytics.user_segmentation.pred_online_casino pred
# MAGIC inner join      uat_analytics.user_segmentation.cluster_online_casino c on c.cluster_id = pred.cluster_id
# MAGIC inner join      prod_silver.user.user_profile up on pred.unique_user_id = concat(up.source_system, "-", up.user_id)
# MAGIC left join       prod_silver.reference_data.country_region_info cri on cri.id = up.nation_id
# MAGIC where           cri.country is not null
# MAGIC group by all
# MAGIC ''')
# MAGIC
# MAGIC pdf = df.toPandas()
# MAGIC
# MAGIC # Chaves de cluster (para dropdown)
# MAGIC clusters = (
# MAGIC     pdf[["cluster_id", "cluster_name"]]
# MAGIC     .drop_duplicates()
# MAGIC     .sort_values(["cluster_id"])
# MAGIC     .to_dict("records")
# MAGIC )
# MAGIC
# MAGIC # Cria 1 trace por cluster
# MAGIC fig = go.Figure()
# MAGIC for i, c in enumerate(clusters):
# MAGIC     cluster_id = c["cluster_id"]
# MAGIC     cluster_name = c["cluster_name"]
# MAGIC     sub = pdf[pdf["cluster_id"] == cluster_id]
# MAGIC
# MAGIC     fig.add_trace(
# MAGIC         go.Choropleth(
# MAGIC             locations=sub["country"],                  # nomes dos países
# MAGIC             locationmode="country names",              # importante
# MAGIC             z=sub["user_count"],
# MAGIC             text=sub["country"],
# MAGIC             name=f"{cluster_id} - {cluster_name}",
# MAGIC             visible=(i == 0),
# MAGIC             hovertemplate=(
# MAGIC                 "<b>%{text}</b><br>"
# MAGIC                 "Users: %{z:,}<extra></extra>"
# MAGIC             ),
# MAGIC             colorbar_title="Users",
# MAGIC         )
# MAGIC     )
# MAGIC
# MAGIC # Botões do dropdown (alternam quais traces ficam visíveis)
# MAGIC buttons = []
# MAGIC n = len(clusters)
# MAGIC for i, c in enumerate(clusters):
# MAGIC     vis = [False] * n
# MAGIC     vis[i] = True
# MAGIC     label = f"{c['cluster_id']} - {c['cluster_name']}"
# MAGIC     buttons.append(
# MAGIC         dict(
# MAGIC             label=label,
# MAGIC             method="update",
# MAGIC             args=[
# MAGIC                 {"visible": vis},
# MAGIC                 {"title": f"Users by Country — {label}"}
# MAGIC             ],
# MAGIC         )
# MAGIC     )
# MAGIC
# MAGIC fig.update_layout(
# MAGIC     title=f"Users by Country — {clusters[0]['cluster_id']} - {clusters[0]['cluster_name']}",
# MAGIC     geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
# MAGIC     updatemenus=[
# MAGIC         dict(
# MAGIC             buttons=buttons,
# MAGIC             direction="down",
# MAGIC             x=0.01, xanchor="left",
# MAGIC             y=1.08, yanchor="top",
# MAGIC             showactive=True,
# MAGIC         )
# MAGIC     ],
# MAGIC     margin=dict(l=0, r=0, t=80, b=0),
# MAGIC )
# MAGIC
# MAGIC fig.show()

# COMMAND ----------

# MAGIC %skip
# MAGIC # INSTALL DEPENDENCIES TO SAVE THE REPORT FILES
# MAGIC
# MAGIC # %pip install playwright
# MAGIC # %pip install nest_asyncio
# MAGIC # !playwright install chromium
# MAGIC
# MAGIC pdf_path  = "/Volumes/uat_gold/machine_learning/segmentation_reports/customer_segmentation_online_casino.pdf"
# MAGIC
# MAGIC import nest_asyncio
# MAGIC import asyncio
# MAGIC from playwright.async_api import async_playwright
# MAGIC
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC async def html_to_pdf(html_path, pdf_path):
# MAGIC     async with async_playwright() as p:
# MAGIC         browser = await p.chromium.launch()
# MAGIC         page = await browser.new_page(viewport={"width": 1280, "height": 900})
# MAGIC         await page.goto(f"file://{html_path}")
# MAGIC         await page.pdf(
# MAGIC             path=pdf_path,
# MAGIC             format="A4",
# MAGIC             print_background=True,
# MAGIC             margin={
# MAGIC                 "top": "20mm",
# MAGIC                 "bottom": "20mm",
# MAGIC                 "left": "15mm",
# MAGIC                 "right": "15mm"
# MAGIC             }
# MAGIC         )
# MAGIC         await browser.close()
# MAGIC
# MAGIC asyncio.run(html_to_pdf(html_path, pdf_path))
# MAGIC
# MAGIC pdf_path