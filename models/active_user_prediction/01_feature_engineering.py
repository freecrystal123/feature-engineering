# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 – Feature Engineering (Active User Prediction)
# MAGIC
# MAGIC Builds per-user features from `fact_wager` to power churn prediction.
# MAGIC
# MAGIC **Key definitions:**
# MAGIC - **Active user last week** – placed ≥ 1 bet during the previous ISO week (Mon–Sun)
# MAGIC - **Churn user this week** – was active last week but placed 0 bets this week
# MAGIC - **Prediction target** – `label = 0` (churn) / `label = 1` (retained)
# MAGIC
# MAGIC For each historical reference week W the notebook:
# MAGIC 1. Identifies users active in week W-1 (candidate set)
# MAGIC 2. Labels them: did they bet in week W?
# MAGIC 3. Builds features from wagering history up to the end of week W-1

# COMMAND ----------

# MAGIC %run ./00_config_active_user_prediction

# COMMAND ----------

from pyspark.sql import functions as F, Window

# COMMAND ----------

dbutils.widgets.text("analysis_id", "analysis_4")

# COMMAND ----------

ANALYSIS_ID     = dbutils.widgets.get("analysis_id").strip()
ANALYSIS_NAME   = ANALYSIS_NAMES[ANALYSIS_ID]
ANALYSIS_FILTER = ANALYSIS_FILTERS[ANALYSIS_ID]

# CURRENT_DATE comes from the global 00_config as F.current_date()
current_date = CURRENT_DATE

# Week boundaries used throughout
# Last week: the full ISO week before the current week
last_week_start = F.date_trunc("week", F.date_sub(current_date, 7))
last_week_end   = F.date_sub(F.date_trunc("week", current_date), 1)  # last Sunday

# COMMAND ----------

wager        = spark.table(FACT_WAGER)
user         = spark.table(DIM_USER)
wager_history = spark.table(T_WAGER_HISTORY).select("id", "wager_winning_status")

wager_time_col   = "wager_placement_timestamp"
wager_stake_col  = "wager_amount"
wager_result_col = "wager_winning_status"

wager = (
    wager
    .withColumnRenamed("source_system", "main_source_system")
    .join(user,          wager.user_sk  == user.user_sk,          "left")
    .join(wager_history, wager.wager_id == wager_history.id,      "left")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A – Active cohort: users who placed ≥ 1 bet last week

# COMMAND ----------

def get_active_users_last_week():
    return (
        wager
        .filter(ANALYSIS_FILTER)
        .filter(
            (F.to_date(wager_time_col) >= last_week_start) &
            (F.to_date(wager_time_col) <= last_week_end)
        )
        .select("user_id")
        .distinct()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### B – Labeled training set (historical weeks)
# MAGIC
# MAGIC For each reference week W in the lookback window:
# MAGIC - Candidate set = users active in week W-1
# MAGIC - `label = 1` if the user also bet in week W (retained), `label = 0` (churn)

# COMMAND ----------

def get_labeled_samples(num_weeks: int = TRAINING_WEEKS_LOOKBACK):
    """Return a labeled DataFrame for supervised churn model training."""
    labeled_dfs = []

    for offset in range(1, num_weeks + 1):
        w_end     = F.date_sub(F.date_trunc("week", current_date), offset * 7)
        w_start   = F.date_add(F.date_sub(F.date_trunc("week", current_date), (offset + 1) * 7), 0)
        wp1_end   = F.date_sub(w_start, 1)
        wp1_start = F.date_sub(wp1_end, 6)

        active_wp1 = (
            wager
            .filter(ANALYSIS_FILTER)
            .filter(
                (F.to_date(wager_time_col) >= wp1_start) &
                (F.to_date(wager_time_col) <= wp1_end)
            )
            .select("user_id")
            .distinct()
            .withColumn("fold", F.lit(offset))
        )

        active_w = (
            wager
            .filter(ANALYSIS_FILTER)
            .filter(
                (F.to_date(wager_time_col) >= w_start) &
                (F.to_date(wager_time_col) <= w_end)
            )
            .select("user_id")
            .distinct()
            .withColumn("label", F.lit(1))
        )

        fold_df = (
            active_wp1
            .join(active_w, "user_id", "left")
            .withColumn("label", F.coalesce(F.col("label"), F.lit(0)))
            .withColumn("ref_week_end", wp1_end)
        )
        labeled_dfs.append(fold_df)

    from functools import reduce
    return reduce(lambda a, b: a.union(b), labeled_dfs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C – Per-user features (as of a given snapshot date)

# COMMAND ----------

def get_features_for_snapshot(snapshot_date):
    """
    Build engagement, recency, and frequency features for every user
    whose wagering history ends on `snapshot_date`.
    """
    hist_start   = F.date_sub(snapshot_date, 365)
    recent_start = F.date_sub(snapshot_date, 90)

    wager_hist = (
        wager
        .filter(ANALYSIS_FILTER)
        .filter(F.to_date(wager_time_col).between(hist_start, snapshot_date))
    )

    feat_engagement = (
        wager_hist
        .groupBy("user_id")
        .agg(
            F.count("*").alias("num_bets_total"),
            F.countDistinct(F.to_date(wager_time_col)).alias("num_active_days"),
            F.countDistinct(
                F.concat_ws("-", F.year(wager_time_col), F.weekofyear(wager_time_col))
            ).alias("num_active_weeks"),
            F.min(wager_time_col).alias("first_bet_time"),
            F.max(wager_time_col).alias("last_bet_time"),
        )
        .withColumn("days_since_last_bet", F.datediff(snapshot_date, F.to_date("last_bet_time")))
        .withColumn(
            "avg_bets_per_active_day",
            F.when(F.col("num_active_days") > 0,
                   F.col("num_bets_total") / F.col("num_active_days"))
            .otherwise(0.0)
        )
    )

    feat_recent = (
        wager_hist
        .filter(F.to_date(wager_time_col) >= recent_start)
        .groupBy("user_id")
        .agg(
            F.count("*").alias("num_bets_90d"),
            F.countDistinct(F.to_date(wager_time_col)).alias("active_days_90d"),
            F.countDistinct(
                F.concat_ws("-", F.year(wager_time_col), F.weekofyear(wager_time_col))
            ).alias("active_weeks_90d"),
            F.sum(wager_stake_col).alias("stake_sum_90d"),
            F.avg(wager_stake_col).alias("stake_avg_90d"),
        )
    )

    recent_4w  = F.date_sub(snapshot_date, 28)
    prior_4w_s = F.date_sub(snapshot_date, 56)
    prior_4w_e = F.date_sub(snapshot_date, 29)

    bets_last_4w = (
        wager_hist
        .filter(F.to_date(wager_time_col).between(recent_4w, snapshot_date))
        .groupBy("user_id")
        .agg(F.count("*").alias("bets_last_4w"))
    )
    bets_prior_4w = (
        wager_hist
        .filter(F.to_date(wager_time_col).between(prior_4w_s, prior_4w_e))
        .groupBy("user_id")
        .agg(F.count("*").alias("bets_prior_4w"))
    )
    feat_trend = (
        bets_last_4w
        .join(bets_prior_4w, "user_id", "outer")
        .fillna({"bets_last_4w": 0, "bets_prior_4w": 0})
        .withColumn(
            "bet_count_trend",
            F.when(F.col("bets_prior_4w") > 0,
                   (F.col("bets_last_4w") - F.col("bets_prior_4w")) / F.col("bets_prior_4w"))
            .otherwise(F.lit(0.0))
        )
    )

    feat_financial = (
        wager_hist
        .groupBy("user_id")
        .agg(
            F.sum(wager_stake_col).alias("stake_sum_total"),
            F.stddev(wager_stake_col).alias("stake_stddev"),
            F.avg(wager_stake_col).alias("stake_avg"),
        )
        .withColumn(
            "stake_cv",
            F.when(F.col("stake_avg") > 0, F.col("stake_stddev") / F.col("stake_avg"))
            .otherwise(0.0)
        )
        .select("user_id", "stake_sum_total", "stake_cv")
    )

    win_condition = F.col(wager_result_col) == F.lit(WAGER_WIN_STATUS)
    feat_winrate = (
        wager_hist
        .groupBy("user_id")
        .agg(
            (F.sum(F.when(win_condition, 1).otherwise(0)) / F.count("*")).alias("win_rate")
        )
    )

    features = (
        feat_engagement
        .join(feat_recent,    "user_id", "left")
        .join(feat_trend,     "user_id", "left")
        .join(feat_financial, "user_id", "left")
        .join(feat_winrate,   "user_id", "left")
        .fillna({
            "num_bets_90d": 0, "active_days_90d": 0, "active_weeks_90d": 0,
            "stake_sum_90d": 0.0, "stake_avg_90d": 0.0,
            "bets_last_4w": 0, "bets_prior_4w": 0, "bet_count_trend": 0.0,
            "stake_sum_total": 0.0, "stake_cv": 0.0,
            "win_rate": 0.0,
        })
    )
    return features

# COMMAND ----------

# MAGIC %md
# MAGIC ### D – Build labeled training features and inference features

# COMMAND ----------

def build_training_features():
    """
    Join historical labels with per-snapshot features to create the training dataset.
    Each row = one user × one historical week fold.
    """
    labeled = get_labeled_samples()
    snapshot_dates = (
        labeled
        .select("ref_week_end")
        .distinct()
        .orderBy("ref_week_end")
        .collect()
    )

    fold_feature_dfs = []
    for row in snapshot_dates:
        snap          = F.lit(row["ref_week_end"])
        fold_labels   = labeled.filter(F.col("ref_week_end") == snap)
        fold_features = get_features_for_snapshot(snap)
        fold_df = (
            fold_labels
            .join(fold_features, "user_id", "left")
            .withColumn("snapshot_date", snap)
        )
        fold_feature_dfs.append(fold_df)

    from functools import reduce
    return reduce(lambda a, b: a.union(b), fold_feature_dfs)


def build_inference_features():
    """
    Features for users who were active last week — scored at prediction time.
    Snapshot date = last_week_end.
    """
    active_cohort = get_active_users_last_week()
    features      = get_features_for_snapshot(last_week_end)
    return (
        active_cohort
        .join(features, "user_id", "left")
        .withColumn("snapshot_date", last_week_end)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### E – Publish feature tables

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

training_features = build_training_features()
publish_result_train = publish_table(
    spark=spark,
    df=training_features.withColumn("created_at", F.current_timestamp()),
    table_base=f"{T_ACTIVE_USER_FEATURES_PREFIX}_{ANALYSIS_NAME}_train",
    analysis_name=f"{ANALYSIS_NAME}_train",
    current_mode="table",
)
print(f"Training features: {publish_result_train['current_table']} (run_id={publish_result_train['run_id']})")

# COMMAND ----------

inference_features = build_inference_features()
publish_result_infer = publish_table(
    spark=spark,
    df=inference_features.withColumn("created_at", F.current_timestamp()),
    table_base=f"{T_ACTIVE_USER_FEATURES_PREFIX}_{ANALYSIS_NAME}_infer",
    analysis_name=f"{ANALYSIS_NAME}_infer",
    current_mode="table",
)
print(f"Inference features: {publish_result_infer['current_table']} (run_id={publish_result_infer['run_id']})")
