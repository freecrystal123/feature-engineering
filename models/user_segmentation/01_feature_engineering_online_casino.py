# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 - feature engineering (online casino)

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import NumericType
from pyspark.sql.functions import broadcast

# COMMAND ----------

# Parameters
GENERAL_DAYS_BEFORE = 365 # days
DAYS_RECENT = 90


ANALYSIS_ID = 'analysis_1'
ANALYSIS_NAME = 'online_casino'
ANALYSIS_FILTER = ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(1, 99)))

if CURRENT_DATE is None:
    current_date = F.current_date()


recent_start = F.date_sub(current_date, DAYS_RECENT)
first_start_analysis = F.date_sub(current_date, GENERAL_DAYS_BEFORE)


# COMMAND ----------

wager = spark.table(FACT_WAGER)
dim_wager = spark.table(DIM_WAGER)
deposit = spark.table(FACT_DEPOSIT)
withdraw = spark.table(FACT_WITHDRAWAL)
user = spark.table(DIM_USER)
dim_coupon = spark.table(DIM_COUPON)
dim_game = spark.table(DIM_GAME)

wager_time_col = "wager_placement_timestamp"
wager_stake_col = "wager_amount"
wager_winning_status_id_col = "wager_winning_status_id"

deposit_time_col = "deposit_timestamp"
deposit_amount_col = "deposit_amount_usd"

withdraw_time_col = "withdrawal_timestamp"
withdraw_amount_col = "withdrawal_amount_usd"

coupon_valid_end_date_col = "valid_end_date"



wager = (
    wager
        .filter(F.to_date(wager_time_col) >= first_start_analysis)
        .withColumnRenamed("source_system", "main_source_system")
        .withColumnRenamed("coupon_id", "wager_coupon_id")
        .join(
            user,
            wager.user_sk == user.user_sk,
            "left"
        )
        .join(
            dim_wager,
            wager.wager_sk == dim_wager.wager_sk,
            "left"
        )
        .join(
            dim_coupon,
            wager.coupon_sk == dim_coupon.coupon_sk,
            "left"
        )
        .join(
            dim_game,
            wager.game_sk == dim_game.game_sk,
            "left"
        )
)
deposit = (
    deposit
        .filter(F.to_date(deposit_time_col) >= first_start_analysis)
        .withColumnRenamed("source_system", "main_source_system")
        .withColumn("game_category_id", F.lit(99))
        .join(
            user,
            deposit.user_sk == user.user_sk,
            "inner"
        )
)
withdraw = (
    withdraw
        .filter(F.to_date(withdraw_time_col) >= first_start_analysis)
        .withColumnRenamed("source_system", "main_source_system")
        .withColumn("game_category_id", F.lit(99))
        .join(
            user,
            withdraw.user_sk == user.user_sk,
            "inner"
        )
)


# if ENVIRONMENT == 'dev': # limiting few users for the analysis
#     wager = wager.orderBy(F.col(wager_time_col).desc()).limit(100)
#     few_user_ids = wager.select('user_id').distinct()
    
#     deposit = deposit.join(broadcast(few_user_ids), "user_id", "semi")
#     withdraw = withdraw.join(broadcast(few_user_ids), "user_id", "semi")
#     user_profile = user_profile.join(broadcast(few_user_ids), "user_id", "semi")
#     coupon = coupon.join(broadcast(few_user_ids.withColumnRenamed("user_id", "uid")).alias("uid"), "uid", "semi")

# display(wager)

# COMMAND ----------

# MAGIC %skip
# MAGIC analysis_def_filters = {
# MAGIC     "analysis_1": (F.col("main_source_system") == "LOTTERY"),                                               # Lottery
# MAGIC     "analysis_2": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(3, 99))),   # Live Casino
# MAGIC     "analysis_3": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(1, 99))),   # Online Casino
# MAGIC     "analysis_4": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(6, 99))),   # Sports
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC ### A - Engagement (fact_wager)

# COMMAND ----------

def get_features_a(analysis:str):

    # Compute engagement metrics
    w_win = Window.partitionBy("user_id").orderBy(F.col('event_ts').cast("timestamp"))

    wager_intervals = (
        wager
        .filter(ANALYSIS_FILTER)
        .select("user_id", F.col(wager_time_col).cast("timestamp").alias("event_ts"))
        .withColumn("prev_ts", F.lag("event_ts").over(w_win))
        .withColumn("diff_hours",
            (F.col("event_ts").cast("long") - F.col("prev_ts").cast("long")) / 3600.0
        )
    )

    engagement_all = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            F.count("*").alias("num_bets_total"),
            F.countDistinct(F.to_date(wager_time_col)).alias("num_active_days"),
            F.min(wager_time_col).alias("first_bet_time"),
            F.max(wager_time_col).alias("last_bet_time"),
        )
        .withColumn("avg_bets_per_day", F.when(F.col("num_active_days") > 0,
                                            F.col("num_bets_total") / F.col("num_active_days")).otherwise(0.0))
        .withColumn("days_since_last_bet", F.datediff(current_date, F.to_date("last_bet_time")))
    )

    engagement_recent_weeks = (
        wager
        .filter(F.to_date(wager_time_col) >= recent_start)
        .filter(ANALYSIS_FILTER)
        .withColumn("year_week", F.concat_ws("-", F.year(wager_time_col), F.weekofyear(wager_time_col)))
        .groupBy("user_id")
        .agg(F.countDistinct("year_week").alias("active_weeks_3m"))
    )

    engagement_intervals = (
        wager_intervals.groupBy("user_id").agg(F.avg("diff_hours").alias("avg_time_between_bets"))
    )

    features_A = (
        engagement_all
        .join(engagement_recent_weeks, "user_id", "left")
        .join(engagement_intervals, "user_id", "left")
        .fillna({"active_weeks_3m": 0})
    )

    return features_A

# features_A.createOrReplaceTempView("features_A")

# features_A = get_features_a("analysis_1")
# display(features_A)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B - Financial (fact_wager + fact_deposit + fact_withdrawal)

# COMMAND ----------

def get_features_b(analysis:str):

    # Base financial from wager
    finance_wager = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            F.sum(F.col(wager_stake_col)).alias("total_stake_amount"),
            F.avg(F.col(wager_stake_col)).alias("avg_stake_amount")
        )
    )

    # Deposits
    finance_deposit = (
        deposit
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            F.sum(deposit_amount_col).alias("total_deposit_amount"),
            F.max(deposit_time_col).alias("last_deposit_time"),
            F.countDistinct(F.to_date(deposit_time_col)).alias("deposit_days_count")
        )
        .withColumn("days_since_last_deposit",
                    F.datediff(current_date, F.to_date("last_deposit_time")))
    )

    # Withdrawals

    finance_withdraw = (
        withdraw
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(F.sum(withdraw_amount_col).alias("total_withdraw_amount"))
    )

    # Combine
    features_B = (
        finance_wager
        .join(finance_deposit, "user_id", "left")
        .join(finance_withdraw, "user_id", "left")
        .fillna({"total_withdraw_amount": 0.0, "total_deposit_amount": 0.0})
        .withColumn("net_cash_flow", F.col("total_deposit_amount") - F.col("total_withdraw_amount"))
    )

    return features_B

# features_B.createOrReplaceTempView("features_B")

# features_B = get_features_b("analysis_2")
# display(features_B)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C - Risk / Performance (fact_wager)

# COMMAND ----------

def get_features_c(analysis:str):

    # ==== 1. Win/Loss Rate ========================================================
    win_condition = F.col("wager_winning_status_id") == F.lit(3) 
    wager_winloss = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            (F.sum(F.when(win_condition, 1).otherwise(0)) / F.count("*")).alias("win_rate")
        )
        .withColumn("loss_rate", 1 - F.col("win_rate"))
    )

    # ==== 2. Stake Volatility =====================================================
    # Standart deviation stakes / average stakes
    stake_volatility = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            F.stddev(F.col(wager_stake_col)).alias("stake_stddev"),
            F.avg(F.col(wager_stake_col)).alias("stake_avg")
        )
        .withColumn("stake_stddev_ratio",
                    F.when(F.col("stake_avg") > 0, F.col("stake_stddev") / F.col("stake_avg")).otherwise(0.0))
        .select("user_id", "stake_stddev_ratio")
    )

    # ==== 3. Stake-to-Deposit Ratio ==============================================
    total_stake = wager.filter(ANALYSIS_FILTER).groupBy("user_id").agg(F.sum(F.col(wager_stake_col)).alias("total_stake_amount"))
    total_deposit = deposit.filter(ANALYSIS_FILTER).groupBy("user_id").agg(F.sum(deposit_amount_col).alias("total_deposit_amount"))
    stake_to_deposit = (
        total_stake.join(total_deposit, "user_id", "left")
        .withColumn(
            "stake_to_deposit_ratio",
            F.when(F.col("total_deposit_amount") > 0,
                    F.col("total_stake_amount") / F.col("total_deposit_amount"))
                .otherwise(F.lit(0.0))
        )
        .select("user_id", "stake_to_deposit_ratio")
    )

    # ==== 4. Net Profit Ratio  ======================
    # status == 3 → win
    win_condition = F.col(wager_winning_status_id_col) == F.lit(3)

    net_profit = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id")
        .agg(
            F.sum(F.when(win_condition, F.col(wager_stake_col)).otherwise(0.0)).alias("sum_wins"),
            F.sum(F.when(~win_condition, F.col(wager_stake_col)).otherwise(0.0)).alias("sum_losses"),
            F.sum(F.col(wager_stake_col)).alias("total_stake")
        )
        .withColumn("net_profit", F.col("sum_wins") - F.col("sum_losses"))
        .withColumn(
            "net_profit_ratio",
            F.when(F.col("total_stake") > 0,
                    F.col("net_profit") / F.col("total_stake"))
                .otherwise(0.0)
        )
        .select("user_id", "net_profit_ratio")
    )

    # ==== 5. Chasing Behavior (increase stake up after losing) =======================
    # status == 3 → win
    w_win = Window.partitionBy("user_id").orderBy(wager_time_col)

    wager_chasing = (
        wager
        .filter(ANALYSIS_FILTER)
        .withColumn("prev_result", F.lag(wager_winning_status_id_col).over(w_win))
        .withColumn("prev_stake", F.lag(F.col(wager_stake_col)).over(w_win))
        .withColumn(
            "chasing_flag",
            F.when(
                (F.col("prev_result") != F.lit(3)) &  # last bet was loss
                (F.col(wager_stake_col) > F.col("prev_stake")), 1  # and increased the stake up
            ).otherwise(0)
        )
        .groupBy("user_id")
        .agg(F.avg("chasing_flag").alias("chasing_ratio"))
    )

    # ==== 6. Max Loss Streak ======================================================
    w = Window.partitionBy("user_id").orderBy(wager_time_col)

    wager_streaks = (
        wager
        .filter(ANALYSIS_FILTER)
        # 1 for loss, 0 to win
        .withColumn("is_loss", F.when(F.col(wager_winning_status_id_col) != F.lit(3), 1).otherwise(0))
        # Creates a new group always there is a victory (is_loss == 0)
        .withColumn("group_id", F.sum(F.when(F.col("is_loss") == 0, 1).otherwise(0)).over(w))
        # Count the loss streaks size inside each group
        .groupBy("user_id", "group_id")
        .agg(F.sum("is_loss").alias("streak_len"))
        # Select the biggest streak
        .groupBy("user_id")
        .agg(F.max("streak_len").alias("max_loss_streak"))
    )

    # ==== 7. Interval Volatility ==================================================
    w = Window.partitionBy("user_id").orderBy(F.col(wager_time_col).cast("timestamp"))

    wager_intervals = (
        wager
        .filter(ANALYSIS_FILTER)
        .select("user_id", F.col(wager_time_col).cast("timestamp"))
        .withColumn("prev_ts", F.lag(F.col(wager_time_col)).over(w))
        .withColumn(
            "diff_hours",
            (F.col(wager_time_col).cast("long") - F.col("prev_ts").cast("long")) / 3600.0
        )
    )
    interval_volatility = (
        wager_intervals.groupBy("user_id")
        .agg(
            F.stddev("diff_hours").alias("interval_stddev"),
            F.avg("diff_hours").alias("interval_avg")
        )
        .withColumn(
            "interval_volatility",
            F.when(F.col("interval_avg") > 0, F.col("interval_stddev") / F.col("interval_avg")).otherwise(0.0)
        )
        .select("user_id", "interval_volatility")
    )



    # ==== 8. Consolidate all metrics ==============================================
    features_C = (
        wager_winloss
        .join(stake_volatility, "user_id", "left")
        .join(stake_to_deposit, "user_id", "left")
        .join(net_profit, "user_id", "left")
        .join(wager_chasing, "user_id", "left")
        .join(wager_streaks, "user_id", "left")
        .join(interval_volatility, "user_id", "left")
        .fillna({
            "win_rate": 0.0, "loss_rate": 0.0,
            "stake_stddev_ratio": 0.0, "stake_to_deposit_ratio": 0.0,
            "net_profit_ratio": 0.0, "chasing_ratio": 0.0,
            "max_loss_streak": 0.0, "interval_volatility": 0.0
        })
        # .withColumn(
        #     "risk_score",
        #     0.25 * F.col("stake_stddev_ratio") +
        #     0.20 * F.col("chasing_ratio") +
        #     0.15 * F.col("interval_volatility") +
        #     0.20 * F.col("loss_rate") +
        #     0.20 * F.col("max_loss_streak")
        # )
    )

    return features_C
    
# features_C.createOrReplaceTempView("features_C")

# features_C = get_features_c("analysis_2")
# display(features_C)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D - RFM (Recency, Frequency, Monetary) – fact_wager

# COMMAND ----------

def get_features_d(analysis: str):
        
    rfm = (
        wager
        .filter(ANALYSIS_FILTER)
        .filter(F.to_date(wager_time_col) >= recent_start)
        .groupBy("user_id")
        .agg(
            F.max(wager_time_col).alias("last_bet_time_90d"),
            F.countDistinct(F.to_date(wager_time_col)).alias("bet_days_90d"),
            F.sum(F.col(wager_stake_col)).alias("stake_sum_90d")
        )
        .withColumn("R_days", F.datediff(current_date, F.to_date("last_bet_time_90d")))
    )

    # Normalize 0–1
    stats = rfm.agg(
        F.min("R_days").alias("R_min"), F.max("R_days").alias("R_max"),
        F.min("bet_days_90d").alias("F_min"), F.max("bet_days_90d").alias("F_max"),
        F.min("stake_sum_90d").alias("M_min"), F.max("stake_sum_90d").alias("M_max")
    ).collect()[0]

    R_min, R_max = stats["R_min"], stats["R_max"]
    F_min, F_max = stats["F_min"], stats["F_max"]
    M_min, M_max = stats["M_min"], stats["M_max"]

    eps = 1e-9

    features_D = (
        rfm
        .withColumn("R_norm", (F.lit(R_max) - F.col("R_days")) / (F.lit(R_max) - F.lit(R_min) + eps))
        .withColumn("F_norm", (F.col("bet_days_90d") - F.lit(F_min)) / (F.lit(F_max) - F.lit(F_min) + eps))
        .withColumn("M_norm", (F.col("stake_sum_90d") - F.lit(M_min)) / (F.lit(M_max) - F.lit(M_min) + eps))
        .withColumn("rfm_score", 0.4*F.col("R_norm") + 0.3*F.col("F_norm") + 0.3*F.col("M_norm"))
    )

    return features_D

# features_D.createOrReplaceTempView("features_D")

# features_D = get_features_d("analysis_2")
# display(features_D)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E - Context / Profile (user_profile + wallet.coupon)

# COMMAND ----------

def get_features_e(analysis: str):
    coupon_features = (
        wager
            .filter(ANALYSIS_FILTER)
            .groupBy("user_id")
            .agg(
                F.count_distinct("coupon_id").alias("num_coupons"),
                F.max(coupon_valid_end_date_col).alias("last_coupon_time")
            )
    )

    features_E = coupon_features.fillna({"num_coupons": 0})

    return features_E

# features_E.createOrReplaceTempView("features_E")

# features_E = get_features_e("analysis_2")
# display(features_E)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine A–E into a single features table

# COMMAND ----------

def get_all_features(analysis: str):
    features_A = get_features_a(analysis)
    features_B = get_features_b(analysis)
    features_C = get_features_c(analysis)
    features_D = get_features_d(analysis)
    features_E = get_features_e(analysis)

    features = (
        features_A
        .join(features_B, "user_id", "left")
        .join(features_C, "user_id", "left")
        .join(features_D, "user_id", "left")
        .join(features_E, "user_id", "left")
        .dropDuplicates(["user_id"])
    )

    return features

# features = get_all_features("analysis_2")
# features.persist()
# display(features.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to Analytics catalog

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

features = get_all_features(ANALYSIS_ID)
(
    features
    .withColumn("created_at", F.current_timestamp())
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{ANALYSIS_NAME}")
)

# for analysis, analysis_name in ANALYSIS_NAMES.items():
#     features = get_all_features(analysis)
#     (
#         features
#         .withColumn("created_at", F.current_timestamp())
#         .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
#         .saveAsTable(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}")
#     )