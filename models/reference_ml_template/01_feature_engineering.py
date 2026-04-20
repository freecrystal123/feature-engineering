# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 - Reference feature engineering

# COMMAND ----------

# MAGIC %run ./00_config_reference_ml_template

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
analysis_id = dbutils.widgets.get("analysis_id").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _pick_col(df, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Column not found for {label}. Tried {candidates}. Available: {df.columns}")


# COMMAND ----------

wager = spark.table(FACT_WAGER)
deposit = spark.table(FACT_DEPOSIT)
users = spark.table(DIM_USER)

user_sk_col = _pick_col(users, ["user_sk"], "user_sk")
user_source_col = _pick_col(users, ["source_system"], "user source_system")
user_id_col = _pick_col(users, ["user_id"], "user_id")
is_test_user_col = _pick_col(users, ["is_test_user"], "is_test_user")

wager_user_sk_col = _pick_col(wager, ["user_sk"], "wager user_sk")
wager_source_col = _pick_col(wager, ["source_system", "main_source_system"], "wager source_system")
wager_ts_col = _pick_col(
    wager,
    ["wager_placement_timestamp", "created_at", "updated_at"],
    "wager timestamp",
)
wager_amount_col = _pick_col(wager, ["wager_amount", "amount"], "wager amount")

dep_user_sk_col = _pick_col(deposit, ["user_sk"], "deposit user_sk")
dep_source_col = _pick_col(deposit, ["source_system", "main_source_system"], "deposit source_system")
dep_ts_col = _pick_col(
    deposit,
    ["deposit_timestamp", "created_at", "updated_at"],
    "deposit timestamp",
)
dep_amount_col = _pick_col(deposit, ["deposit_amount_usd", "deposit_amount", "amount"], "deposit amount")


# COMMAND ----------

current_date = CURRENT_DATE
features_start = F.date_sub(current_date, F.lit(int(FEATURE_LOOKBACK_DAYS)))
label_start = F.date_sub(current_date, F.lit(int(LABEL_WINDOW_DAYS)))


valid_users = (
    users
    .filter(F.col(is_test_user_col) == F.lit(False))
    .select(
        F.col(user_sk_col).alias("user_sk"),
        F.col(user_source_col).alias("source_system"),
        F.col(user_id_col).cast("string").alias("user_id"),
    )
)

wager_base = (
    wager.alias("w")
    .join(
        valid_users.alias("u"),
        (F.col(f"w.{wager_user_sk_col}") == F.col("u.user_sk"))
        & (F.col(f"w.{wager_source_col}") == F.col("u.source_system")),
        "inner",
    )
    .filter(F.col(f"w.{wager_source_col}") != F.lit("LOTTERY"))
    .select(
        F.col("u.source_system"),
        F.col("u.user_id"),
        F.to_timestamp(F.col(f"w.{wager_ts_col}")).alias("event_ts"),
        F.coalesce(F.col(f"w.{wager_amount_col}").cast("double"), F.lit(0.0)).alias("wager_amount"),
    )
    .filter(F.col("event_ts").isNotNull())
)

deposit_base = (
    deposit.alias("d")
    .join(
        valid_users.alias("u"),
        (F.col(f"d.{dep_user_sk_col}") == F.col("u.user_sk"))
        & (F.col(f"d.{dep_source_col}") == F.col("u.source_system")),
        "inner",
    )
    .filter(F.col(f"d.{dep_source_col}") != F.lit("LOTTERY"))
    .select(
        F.col("u.source_system"),
        F.col("u.user_id"),
        F.to_timestamp(F.col(f"d.{dep_ts_col}")).alias("event_ts"),
        F.coalesce(F.col(f"d.{dep_amount_col}").cast("double"), F.lit(0.0)).alias("deposit_amount"),
    )
    .filter(F.col("event_ts").isNotNull())
)


# COMMAND ----------

wager_features = (
    wager_base
    .filter(F.to_date("event_ts") >= features_start)
    .groupBy("source_system", "user_id")
    .agg(
        F.count(F.lit(1)).alias("num_bets_90d"),
        F.sum("wager_amount").alias("total_stake_90d"),
        F.avg("wager_amount").alias("avg_stake_90d"),
        F.max("event_ts").alias("last_bet_ts"),
    )
    .withColumn("days_since_last_bet", F.datediff(current_date, F.to_date("last_bet_ts")))
)

deposit_features = (
    deposit_base
    .filter(F.to_date("event_ts") >= features_start)
    .groupBy("source_system", "user_id")
    .agg(
        F.count(F.lit(1)).alias("num_deposits_90d"),
        F.sum("deposit_amount").alias("total_deposit_90d"),
    )
)

label_df = (
    wager_base
    .filter(F.to_date("event_ts") >= label_start)
    .groupBy("source_system", "user_id")
    .agg(F.count(F.lit(1)).alias("num_bets_7d"))
    .withColumn("label_active_7d", (F.col("num_bets_7d") > 0).cast("int"))
    .drop("num_bets_7d")
)


# COMMAND ----------

features = (
    valid_users
    .join(wager_features, on=["source_system", "user_id"], how="left")
    .join(deposit_features, on=["source_system", "user_id"], how="left")
    .join(label_df, on=["source_system", "user_id"], how="left")
    .fillna(
        {
            "num_bets_90d": 0,
            "total_stake_90d": 0.0,
            "avg_stake_90d": 0.0,
            "days_since_last_bet": int(FEATURE_LOOKBACK_DAYS),
            "num_deposits_90d": 0,
            "total_deposit_90d": 0.0,
            "label_active_7d": 0,
        }
    )
    .withColumn("unique_user_id", F.concat_ws("-", F.col("source_system"), F.col("user_id")))
    .filter(F.col("unique_user_id").isNotNull())
    .drop("user_sk", "source_system", "user_id", "last_bet_ts")
)


# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    published = publish_table(
        spark=spark,
        df=features.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_REFERENCE_FEATURES_PREFIX}_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
    )
    print(
        f"Published features: {published['snapshot_table']} and {published['current_table']} "
        f"(run_id={published['run_id']})"
    )
