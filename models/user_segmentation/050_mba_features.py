# Databricks notebook source
# MAGIC %md
# MAGIC ## 5.0 - MBA features (Casino + Sports)
# MAGIC
# MAGIC Goal:
# MAGIC - Build user-item baskets for Market Basket Analysis (cross-platform behavior).
# MAGIC - Use wager facts (`fact_settled_wager`) as the single source of wagers/events.
# MAGIC - For Sports item names, enrich from `prod_silver.wager.wager_history` +
# MAGIC   `prod_creative.shared.dim_sport_type` (`sport_type = sport_tyoe`).
# MAGIC - For Casino item names, use `dim_game.game_name`.

# COMMAND ----------

# MAGIC %run ./050_config_mba

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame

# COMMAND ----------

settled_wager = spark.table(T_SETTLED_WAGER)
dim_wager = spark.table(DIM_WAGER)
dim_game = spark.table(DIM_GAME)
user = spark.table(T_USER)
wager_history = spark.table("prod_silver.wager.wager_history")
dim_sport_type = spark.table("prod_creative.shared.dim_sport_type")

# COMMAND ----------

def _pick_col(df: DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find column for {label}. Tried {candidates}. Available: {df.columns}")


analysis_name = ANALYSIS_NAMES[MBA_ANALYSIS]
CASINO_CATEGORIES = [1, 3]
SPORTS_CATEGORY = 6
ANALYSIS_CATEGORIES = [1, 3, 6]

settled_source_col = _pick_col(settled_wager, ["source_system", "main_source_system", "source"], "settled source_system")
settled_ts_col = _pick_col(
    settled_wager,
    ["wager_settlement_timestamp", "wager_placement_timestamp", "created_at", "updated_at"],
    "settled wager timestamp",
)
settled_wager_col = _pick_col(settled_wager, ["wager_amount", "amount"], "settled wager amount")
settled_wager_sk_col = _pick_col(settled_wager, ["wager_sk"], "settled wager_sk")

dw_wager_sk_col = _pick_col(dim_wager, ["wager_sk"], "dim_wager wager_sk")
dw_wager_id_col = _pick_col(dim_wager, ["wager_id"], "dim_wager wager_id")
dw_source_col = _pick_col(dim_wager, ["source_system", "main_source_system", "source"], "dim_wager source_system")

wh_id_col = _pick_col(wager_history, ["id"], "wager_history id")
wh_source_col = _pick_col(wager_history, ["source_system", "source"], "wager_history source_system")
wh_sport_type_col = _pick_col(wager_history, ["sport_type", "sport_tyoe"], "wager_history sport_type")

# COMMAND ----------

# Build sports mapping keyed by wager_history PK (id, source_system)
sports_lookup = (
    wager_history.alias("w")
    .select(
        F.col(f"w.{wh_id_col}").cast("string").alias("wh_wager_id"),
        F.col(f"w.{wh_source_col}").cast("string").alias("wh_source_system"),
        F.col(f"w.{wh_sport_type_col}").cast("int").alias("sport_type"),
    )
    .dropna(subset=["wh_wager_id", "wh_source_system"])
    .groupBy("wh_wager_id", "wh_source_system")
    .agg(F.max("sport_type").alias("sport_type"))
    .alias("wh")
)

sports_type_lookup = (
    dim_sport_type
    .select(
        F.col("sport_tyoe").cast("int").alias("sport_type"),
        F.col("sport_name").cast("string").alias("sport_name"),
    )
    .dropDuplicates(["sport_type"])
    .alias("dst")
)

# COMMAND ----------

def build_features_for_window(lookback_days: int):
    lookback_start = F.date_sub(F.current_date(), F.lit(int(lookback_days)))

    base_events = (
        settled_wager.alias("f")
        .join(
            dim_game.select("game_sk", "source_system", "game_category_id", "game_name").alias("g"),
            (F.col("f.game_sk") == F.col("g.game_sk")) & (F.col(f"f.{settled_source_col}") == F.col("g.source_system")),
            "inner",
        )
        .join(
            user.select("user_sk", "source_system", "user_id", "is_test_user").alias("u"),
            (F.col("f.user_sk") == F.col("u.user_sk")) & (F.col(f"f.{settled_source_col}") == F.col("u.source_system")),
            "inner",
        )
        .filter(F.col("u.is_test_user") == F.lit(False))
        .filter(F.upper(F.col(f"f.{settled_source_col}")) != F.lit("LOTTERY"))
        .filter(F.col("g.game_category_id").isin(ANALYSIS_CATEGORIES))
        .withColumn("event_date", F.to_date(F.col(f"f.{settled_ts_col}")))
        .filter(F.col("event_date") >= lookback_start)
        .join(
            dim_wager.select(
                F.col(dw_wager_sk_col).alias("dw_wager_sk"),
                F.col(dw_wager_id_col).cast("string").alias("dw_wager_id"),
                F.col(dw_source_col).cast("string").alias("dw_source_system"),
            ).alias("dw"),
            (F.col(f"f.{settled_wager_sk_col}") == F.col("dw.dw_wager_sk")) &
            (F.col(f"f.{settled_source_col}").cast("string") == F.col("dw.dw_source_system")),
            "left",
        )
        .join(
            sports_lookup,
            (F.col("dw.dw_wager_id") == F.col("wh.wh_wager_id")) &
            (F.col("dw.dw_source_system") == F.col("wh.wh_source_system")),
            "left",
        )
        .join(
            sports_type_lookup,
            F.col("wh.sport_type") == F.col("dst.sport_type"),
            "left",
        )
        .withColumn(
            "vertical",
            F.when(F.col("g.game_category_id") == F.lit(SPORTS_CATEGORY), F.lit("sports")).otherwise(F.lit("casino")),
        )
        .withColumn(
            "item_name",
            F.when(
                F.col("g.game_category_id") == F.lit(SPORTS_CATEGORY),
                F.coalesce(F.col("dst.sport_name"), F.lit("Unknown Sport")),
            ).otherwise(F.coalesce(F.col("g.game_name"), F.lit("Unknown Casino Game"))),
        )
        .select(
            F.col("u.user_id").cast("string").alias("user_id"),
            F.col(f"f.{settled_source_col}").cast("string").alias("source_system"),
            F.col("vertical"),
            F.col("item_name"),
            F.coalesce(F.col(f"f.{settled_wager_col}").cast("double"), F.lit(0.0)).alias("wager_amount"),
            F.col("event_date"),
        )
    )

    all_events = (
        base_events
        .withColumn("item_name_clean", F.trim(F.regexp_replace(F.col("item_name"), r"\s+", " ")))
        .withColumn("item_name_clean", F.when(F.col("item_name_clean") == "", F.lit("Unknown")).otherwise(F.col("item_name_clean")))
        .withColumn("item_id", F.concat_ws("::", F.col("vertical"), F.col("item_name_clean")))
        .withColumn("unique_user_id", F.concat_ws("-", F.col("source_system"), F.col("user_id")))
    )

    user_item_features = (
        all_events
        .groupBy("unique_user_id", "source_system", "user_id", "vertical", "item_id", "item_name_clean")
        .agg(
            F.count(F.lit(1)).alias("num_bets"),
            F.sum(F.coalesce(F.col("wager_amount"), F.lit(0.0))).alias("wager_amount_sum"),
            F.max("event_date").alias("last_event_date"),
        )
        .withColumn(
            "is_item_eligible",
            (
                (F.col("num_bets") >= F.lit(int(MBA_MIN_BETS_PER_ITEM))) |
                (F.col("wager_amount_sum") >= F.lit(float(MBA_MIN_WAGER_PER_ITEM)))
            ).cast("int"),
        )
        .withColumn("lookback_days", F.lit(int(lookback_days)))
    )

    transactions = (
        user_item_features
        .filter(F.col("is_item_eligible") == F.lit(1))
        .groupBy("unique_user_id", "lookback_days")
        .agg(
            F.collect_set("item_id").alias("items"),
            F.sum(F.when(F.col("vertical") == "casino", F.lit(1)).otherwise(F.lit(0))).alias("casino_items_count"),
            F.sum(F.when(F.col("vertical") == "sports", F.lit(1)).otherwise(F.lit(0))).alias("sports_items_count"),
        )
        .withColumn("num_items", F.size("items"))
        .withColumn("is_cross_platform_user", ((F.col("casino_items_count") > 0) & (F.col("sports_items_count") > 0)).cast("int"))
        .filter(F.col("num_items") >= F.lit(int(MBA_MIN_ITEMS_PER_USER)))
    )

    return user_item_features, transactions

# COMMAND ----------

all_user_item_features = None
all_transactions = None

for d in MBA_LOOKBACK_DAYS_LIST:
    ui_df, tx_df = build_features_for_window(int(d))

    if all_user_item_features is None:
        all_user_item_features = ui_df
        all_transactions = tx_df
    else:
        all_user_item_features = all_user_item_features.unionByName(ui_df, allowMissingColumns=True)
        all_transactions = all_transactions.unionByName(tx_df, allowMissingColumns=True)

# COMMAND ----------

(
    all_user_item_features
    .withColumn("analysis_id", F.lit(MBA_ANALYSIS))
    .withColumn("analysis_name", F.lit(analysis_name))
    .withColumn("created_at", F.current_timestamp())
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{T_MBA_USER_ITEM_FEATURES_PREFIX}_{analysis_name}")
)

(
    all_transactions
    .withColumn("analysis_id", F.lit(MBA_ANALYSIS))
    .withColumn("analysis_name", F.lit(analysis_name))
    .withColumn("created_at", F.current_timestamp())
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{T_MBA_TRANSACTIONS_PREFIX}_{analysis_name}")
)

print(f"Saved: {T_MBA_USER_ITEM_FEATURES_PREFIX}_{analysis_name} (lookbacks={MBA_LOOKBACK_DAYS_LIST})")
print(f"Saved: {T_MBA_TRANSACTIONS_PREFIX}_{analysis_name} (lookbacks={MBA_LOOKBACK_DAYS_LIST})")
print(
    "Sports naming join used: "
    f"settled_wager.{settled_wager_sk_col} -> dim_wager.{dw_wager_sk_col} -> "
    f"wager_history.({wh_id_col}, {wh_source_col})"
)

# COMMAND ----------

display(
    spark.table(f"{T_MBA_TRANSACTIONS_PREFIX}_{analysis_name}")
    .groupBy("lookback_days")
    .agg(
        F.countDistinct("unique_user_id").alias("num_users"),
        F.avg("num_items").alias("avg_items"),
        F.avg("is_cross_platform_user").alias("pct_cross_platform_users"),
    )
    .orderBy("lookback_days")
)
