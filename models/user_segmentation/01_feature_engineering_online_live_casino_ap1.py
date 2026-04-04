# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 - feature engineering (online + live casino) - approach 1

# COMMAND ----------

# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import NumericType
from pyspark.sql.functions import broadcast

# COMMAND ----------


ANALYSIS_ID = 'analysis_5'
ANALYSIS_NAME = 'online_live_casino'

# Game category mapping (from your previous notebook notes):
# - Online Casino: game_category_id in (1, 99)
# - Live Casino  : game_category_id in (3, 99)
# Adjust these lists if your taxonomy changes.
# ONLINE_CASINO_CATEGORY_IDS = [1, 99]
# LIVE_CASINO_CATEGORY_IDS   = [3, 99]

# Combined scope = Online Casino + Live Casino (excluding Lottery)
# ONLINE_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*ONLINE_CASINO_CATEGORY_IDS)))
# LIVE_FILTER   = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*LIVE_CASINO_CATEGORY_IDS)))
ANALYSIS_FILTER = ANALYSIS_FILTERS[ANALYSIS_ID]

current_date = CURRENT_DATE

recent_start = F.date_sub(current_date, DAYS_RECENT)
first_start_analysis = F.date_sub(current_date, GENERAL_DAYS_BEFORE)


# COMMAND ----------

wager = spark.table(T_WAGER)
dim_wager = spark.table(DIM_WAGER)
deposit = spark.table(T_DEPOSIT)
withdraw = spark.table(T_WITHDRAW)
user = spark.table(T_USER)
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
        # .withColumnRenamed("source_system", "main_source_system")
        .withColumnRenamed("coupon_id", "wager_coupon_id")
        .join(
            user,
            wager.user_sk == user.user_sk,
            "left"
        ).drop(user.user_sk, user.source_system)
        .join(
            dim_wager,
            wager.wager_sk == dim_wager.wager_sk,
            "left"
        ).drop(dim_wager.wager_sk, dim_wager.source_system)
        .join(
            dim_coupon,
            wager.coupon_sk == dim_coupon.coupon_sk,
            "left"
        ).drop(dim_coupon.coupon_sk, dim_coupon.source_system)
        .join(
            dim_game,
            wager.game_sk == dim_game.game_sk,
            "left"
        ).drop(dim_game.game_sk,dim_game.source_system)
)
deposit = (
    deposit
        .filter(F.to_date(deposit_time_col) >= first_start_analysis)
        # .withColumnRenamed("source_system", "main_source_system")
        .withColumn("game_category_id", F.lit(99))
        .join(
            user,
            deposit.user_sk == user.user_sk,
            "inner"
        ).drop(user.user_sk, user.source_system)
)
withdraw = (
    withdraw
        .filter(F.to_date(withdraw_time_col) >= first_start_analysis)
        # .withColumnRenamed("source_system", "main_source_system")
        .withColumn("game_category_id", F.lit(99))
        .join(
            user,
            withdraw.user_sk == user.user_sk,
            "inner"
        ).drop(user.user_sk, user.source_system)
)


# if ENVIRONMENT == 'dev': # limiting few users for the analysis
#     wager = wager.orderBy(F.col(wager_time_col).desc()).limit(100)
#     few_user_ids = wager.select('user_id').distinct()
    
#     deposit = deposit.join(broadcast(few_user_ids), "user_id", "semi")
#     withdraw = withdraw.join(broadcast(few_user_ids), "user_id", "semi")
#     user_profile = user_profile.join(broadcast(few_user_ids), "user_id", "semi")
#     coupon = coupon.join(broadcast(few_user_ids.withColumnRenamed("user_id", "uid")).alias("uid"), "uid", "semi")

# display(wager)
# display(withdraw)
# display(deposit)

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

def get_features_a(analysis: str):

    def _engagement_block(filter_expr, suffix: str):
        """
        Engagement metrics for a given product scope (filter_expr).
        Returns one row per user_id with suffix-applied columns.
        """
        w_win = Window.partitionBy("user_id", "source_system").orderBy(F.col('event_ts').cast("timestamp"))

        wager_intervals = (
            wager
            .filter(filter_expr)
            .select("user_id", "source_system", F.col(wager_time_col).cast("timestamp").alias("event_ts"))
            .withColumn("prev_ts", F.lag("event_ts").over(w_win))
            .withColumn("diff_hours", (F.col("event_ts").cast("long") - F.col("prev_ts").cast("long")) / 3600.0)
        )
        # display(wager_intervals)

        engagement_all = (
            wager
            .filter(filter_expr)
            .groupBy("user_id", "source_system")
            .agg(
                F.count("*").alias(f"num_bets_total{suffix}"),
                F.countDistinct(F.to_date(wager_time_col)).alias(f"num_active_days{suffix}"),
                F.min(wager_time_col).alias(f"first_bet_time{suffix}"),
                F.max(wager_time_col).alias(f"last_bet_time{suffix}"),
            )
            .withColumn(
                f"avg_bets_per_day{suffix}",
                F.when(F.col(f"num_active_days{suffix}") > 0, F.col(f"num_bets_total{suffix}") / F.col(f"num_active_days{suffix}")).otherwise(0.0)
            )
            .withColumn(
                f"days_since_last_bet{suffix}",
                F.datediff(current_date, F.to_date(F.col(f"last_bet_time{suffix}")))
            )
        )

        engagement_recent_weeks = (
            wager
            .filter(F.to_date(wager_time_col) >= recent_start)
            .filter(filter_expr)
            .withColumn("year_week", F.concat_ws("-", F.year(wager_time_col), F.weekofyear(wager_time_col)))
            .groupBy("user_id", "source_system")
            .agg(F.countDistinct("year_week").alias(f"active_weeks_3m{suffix}"))
        )

        engagement_intervals = (
            wager_intervals.groupBy("user_id", "source_system").agg(F.avg("diff_hours").alias(f"avg_time_between_bets{suffix}"))
        )

        return (
            engagement_all
            .join(engagement_recent_weeks, on=["user_id","source_system"], how="left")
            .join(engagement_intervals, on=["user_id","source_system"], how="left")
            .fillna({f"active_weeks_3m{suffix}": 0})
        )

    # Overall (combined)
    overall = _engagement_block(ANALYSIS_FILTER, "")

    # By product
    online  = _engagement_block(ONLINE_FILTER, "_online")
    live    = _engagement_block(LIVE_FILTER, "_live")

    features_A = (
        overall
        .join(online, on=["user_id","source_system"], how="left")
        .join(live, on=["user_id","source_system"], how="left")
        .fillna({
            "num_bets_total_online": 0,
            "num_active_days_online": 0,
            "avg_bets_per_day_online": 0.0,
            "active_weeks_3m_online": 0,
            "num_bets_total_live": 0,
            "num_active_days_live": 0,
            "avg_bets_per_day_live": 0.0,
            "active_weeks_3m_live": 0,
        })
        # Cross-product preference / mix
        .withColumn(
            "live_share_of_bets",
            F.when(F.col("num_bets_total") > 0, F.col("num_bets_total_live") / F.col("num_bets_total")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_share_of_active_days",
            F.when(F.col("num_active_days") > 0, F.col("num_active_days_live") / F.col("num_active_days")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_vs_online_avg_bets_per_day_ratio",
            F.when(F.col("avg_bets_per_day_online") > 0, F.col("avg_bets_per_day_live") / F.col("avg_bets_per_day_online")).otherwise(F.lit(0.0))
        )
    )

    return features_A

# features_A = get_features_a("analysis_5")
# display(features_A)


# COMMAND ----------

# MAGIC %md
# MAGIC ### B - Financial (fact_wager + fact_deposit + fact_withdrawal)

# COMMAND ----------

def get_features_b(analysis: str):

    def _wager_finance_block(filter_expr, suffix: str):
        return (
            wager
            .filter(filter_expr)
            .groupBy("user_id","source_system")
            .agg(
                F.sum(F.col(wager_stake_col)).alias(f"total_stake_amount{suffix}"),
                F.avg(F.col(wager_stake_col)).alias(f"avg_stake_amount{suffix}"),
                F.count("*").alias(f"num_bets_fin{suffix}")
            )
        )

    # Wager financials (overall + by product)
    finance_wager_all   = _wager_finance_block(ANALYSIS_FILTER, "")
    finance_wager_online= _wager_finance_block(ONLINE_FILTER, "_online")
    finance_wager_live  = _wager_finance_block(LIVE_FILTER, "_live")

    # Deposits (user-level; keep overall)
    finance_deposit = (
        deposit
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id","source_system")
        .agg(
            F.sum(deposit_amount_col).alias("total_deposit_amount"),
            F.max(deposit_time_col).alias("last_deposit_time"),
            F.countDistinct(F.to_date(deposit_time_col)).alias("deposit_days_count")
        )
        .withColumn("days_since_last_deposit", F.datediff(current_date, F.to_date("last_deposit_time")))
    )

    # Withdrawals (user-level; keep overall)
    finance_withdraw = (
        withdraw
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id","source_system")
        .agg(F.sum(withdraw_amount_col).alias("total_withdraw_amount"))
    )

    features_B = (
        finance_wager_all
        .join(finance_wager_online, on=["user_id","source_system"], how="left")
        .join(finance_wager_live, on=["user_id","source_system"], how="left")
        .join(finance_deposit, on=["user_id","source_system"], how="left")
        .join(finance_withdraw, on=["user_id","source_system"], how="left")
        .fillna({
            "total_withdraw_amount": 0.0,
            "total_deposit_amount": 0.0,
            "total_stake_amount_online": 0.0,
            "avg_stake_amount_online": 0.0,
            "num_bets_fin_online": 0,
            "total_stake_amount_live": 0.0,
            "avg_stake_amount_live": 0.0,
            "num_bets_fin_live": 0,
        })
        .withColumn("net_cash_flow", F.col("total_deposit_amount") - F.col("total_withdraw_amount"))
        # Cross-product preference / mix
        .withColumn(
            "live_share_of_total_stake",
            F.when(F.col("total_stake_amount") > 0, F.col("total_stake_amount_live") / F.col("total_stake_amount")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_vs_online_total_stake_ratio",
            F.when(F.col("total_stake_amount_online") > 0, F.col("total_stake_amount_live") / F.col("total_stake_amount_online")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_vs_online_avg_stake_ratio",
            F.when(F.col("avg_stake_amount_online") > 0, F.col("avg_stake_amount_live") / F.col("avg_stake_amount_online")).otherwise(F.lit(0.0))
        )
    )

    return features_B

# features_B = get_features_b("analysis_5")
# display(features_B)


# COMMAND ----------

# MAGIC %md
# MAGIC ### C - Risk / Performance (fact_wager)

# COMMAND ----------

def get_features_c(analysis: str):

    def _risk_block(filter_expr, suffix: str):
        # Win/Loss Rate
        win_condition = F.col(wager_winning_status_id_col) == F.lit(3)

        wager_winloss = (
            wager
            .filter(filter_expr)
            .groupBy("user_id", "source_system")
            .agg((F.sum(F.when(win_condition, 1).otherwise(0)) / F.count("*")).alias(f"win_rate{suffix}"))
            .withColumn(f"loss_rate{suffix}", 1 - F.col(f"win_rate{suffix}"))
        )

        # Stake volatility (std/avg)
        stake_volatility = (
            wager
            .filter(filter_expr)
            .groupBy("user_id","source_system")
            .agg(
                F.stddev(F.col(wager_stake_col)).alias(f"stake_stddev{suffix}"),
                F.avg(F.col(wager_stake_col)).alias(f"stake_avg{suffix}")
            )
            .withColumn(
                f"stake_stddev_ratio{suffix}",
                F.when(F.col(f"stake_avg{suffix}") > 0, F.col(f"stake_stddev{suffix}") / F.col(f"stake_avg{suffix}")).otherwise(0.0)
            )
            .select("user_id", "source_system", f"stake_stddev_ratio{suffix}")
        )

        # Net profit ratio (wins - losses) / total stake
        net_profit = (
            wager
            .filter(filter_expr)
            .groupBy("user_id", "source_system")
            .agg(
                F.sum(F.when(win_condition, F.col(wager_stake_col)).otherwise(0.0)).alias(f"sum_wins{suffix}"),
                F.sum(F.when(~win_condition, F.col(wager_stake_col)).otherwise(0.0)).alias(f"sum_losses{suffix}"),
                F.sum(F.col(wager_stake_col)).alias(f"total_stake{suffix}")
            )
            .withColumn(f"net_profit{suffix}", F.col(f"sum_wins{suffix}") - F.col(f"sum_losses{suffix}"))
            .withColumn(
                f"net_profit_ratio{suffix}",
                F.when(F.col(f"total_stake{suffix}") > 0, F.col(f"net_profit{suffix}") / F.col(f"total_stake{suffix}")).otherwise(0.0)
            )
            .select("user_id", "source_system",f"net_profit_ratio{suffix}")
        )

        # Chasing behavior (increasing stake after a loss)
        w_win = Window.partitionBy("user_id","source_system").orderBy(wager_time_col)
        wager_chasing = (
            wager
            .filter(filter_expr)
            .withColumn("prev_result", F.lag(wager_winning_status_id_col).over(w_win))
            .withColumn("prev_stake", F.lag(F.col(wager_stake_col)).over(w_win))
            .withColumn(
                "chasing_flag",
                F.when((F.col("prev_result") != F.lit(3)) & (F.col(wager_stake_col) > F.col("prev_stake")), 1).otherwise(0)
            )
            .groupBy("user_id","source_system")
            .agg(F.avg("chasing_flag").alias(f"chasing_ratio{suffix}"))
        )

        # Max loss streak
        w = Window.partitionBy("user_id","source_system").orderBy(wager_time_col)
        wager_streaks = (
            wager
            .filter(filter_expr)
            .withColumn("is_loss", F.when(F.col(wager_winning_status_id_col) != F.lit(3), 1).otherwise(0))
            .withColumn("group_id", F.sum(F.when(F.col("is_loss") == 0, 1).otherwise(0)).over(w))
            .groupBy("user_id", "source_system","group_id")
            .agg(F.sum("is_loss").alias("streak_len"))
            .groupBy("user_id","source_system")
            .agg(F.max("streak_len").alias(f"max_loss_streak{suffix}"))
        )

        # Interval volatility (std/avg hours between bets)
        w2 = Window.partitionBy("user_id","source_system").orderBy(F.col('event_ts').cast("timestamp"))
        wager_intervals = (
            wager
            .filter(filter_expr)
            .select("user_id", "source_system", F.col(wager_time_col).cast("timestamp").alias("event_ts"))
            .withColumn("prev_ts", F.lag("event_ts").over(w2))
            .withColumn("diff_hours", (F.col("event_ts").cast("long") - F.col("prev_ts").cast("long")) / 3600.0)
        )
        interval_volatility = (
            wager_intervals.groupBy("user_id","source_system")
            .agg(F.stddev("diff_hours").alias(f"interval_stddev{suffix}"), F.avg("diff_hours").alias(f"interval_avg{suffix}"))
            .withColumn(
                f"interval_volatility{suffix}",
                F.when(F.col(f"interval_avg{suffix}") > 0, F.col(f"interval_stddev{suffix}") / F.col(f"interval_avg{suffix}")).otherwise(0.0)
            )
            .select("user_id", "source_system", f"interval_volatility{suffix}")
        )

        return (
            wager_winloss
            .join(stake_volatility, on=["user_id","source_system"], how="left")
            .join(net_profit, on=["user_id","source_system"], how="left")
            .join(wager_chasing, on=["user_id","source_system"], how="left")
            .join(wager_streaks, on=["user_id","source_system"], how="left")
            .join(interval_volatility, on=["user_id","source_system"], how="left")
            .fillna({
                f"win_rate{suffix}": 0.0,
                f"loss_rate{suffix}": 0.0,
                f"stake_stddev_ratio{suffix}": 0.0,
                f"net_profit_ratio{suffix}": 0.0,
                f"chasing_ratio{suffix}": 0.0,
                f"max_loss_streak{suffix}": 0.0,
                f"interval_volatility{suffix}": 0.0,
            })
        )

    # Overall + product-specific blocks
    risk_all   = _risk_block(ANALYSIS_FILTER, "")
    risk_online= _risk_block(ONLINE_FILTER, "_online")
    risk_live  = _risk_block(LIVE_FILTER, "_live")

    # Stake-to-deposit ratio (overall + by product, deposit is user-level)
    total_deposit = (
        deposit
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id","source_system")
        .agg(F.sum(deposit_amount_col).alias("total_deposit_amount"))
    )

    total_stake_all = wager.filter(ANALYSIS_FILTER).groupBy("user_id","source_system").agg(F.sum(F.col(wager_stake_col)).alias("total_stake_amount"))
    total_stake_online = wager.filter(ONLINE_FILTER).groupBy("user_id","source_system").agg(F.sum(F.col(wager_stake_col)).alias("total_stake_amount_online"))
    total_stake_live = wager.filter(LIVE_FILTER).groupBy("user_id","source_system").agg(F.sum(F.col(wager_stake_col)).alias("total_stake_amount_live"))

    stake_to_deposit_all = (
        total_stake_all.join(total_deposit, on=["user_id","source_system"], how="left")
        .withColumn("stake_to_deposit_ratio",
                    F.when(F.col("total_deposit_amount") > 0, F.col("total_stake_amount") / F.col("total_deposit_amount")).otherwise(F.lit(0.0)))
        .select("user_id", "source_system", "stake_to_deposit_ratio")
    )

    stake_to_deposit_online = (
        total_stake_online.join(total_deposit, on=["user_id","source_system"], how="left")
        .withColumn("stake_to_deposit_ratio_online",
                    F.when(F.col("total_deposit_amount") > 0, F.col("total_stake_amount_online") / F.col("total_deposit_amount")).otherwise(F.lit(0.0)))
        .select("user_id", "source_system", "stake_to_deposit_ratio_online")
    )

    stake_to_deposit_live = (
        total_stake_live.join(total_deposit, on=["user_id","source_system"], how="left")
        .withColumn("stake_to_deposit_ratio_live",
                    F.when(F.col("total_deposit_amount") > 0, F.col("total_stake_amount_live") / F.col("total_deposit_amount")).otherwise(F.lit(0.0)))
        .select("user_id", "source_system", "stake_to_deposit_ratio_live")
    )

    features_C = (
        risk_all
        .join(risk_online, on=["user_id","source_system"], how="left")
        .join(risk_live, on=["user_id","source_system"], how="left")
        .join(stake_to_deposit_all, on=["user_id","source_system"], how="left")
        .join(stake_to_deposit_online, on=["user_id","source_system"], how="left")
        .join(stake_to_deposit_live, on=["user_id","source_system"], how="left")
        .fillna({
            "stake_to_deposit_ratio": 0.0,
            "stake_to_deposit_ratio_online": 0.0,
            "stake_to_deposit_ratio_live": 0.0,
        })
        # Cross-product (risk-style deltas)
        .withColumn(
            "live_vs_online_chasing_ratio",
            F.when(F.col("chasing_ratio_online") > 0, F.col("chasing_ratio_live") / F.col("chasing_ratio_online")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_vs_online_stake_volatility_ratio",
            F.when(F.col("stake_stddev_ratio_online") > 0, F.col("stake_stddev_ratio_live") / F.col("stake_stddev_ratio_online")).otherwise(F.lit(0.0))
        )
    )

    return features_C

# features_C = get_features_c("analysis_5")
# display(features_C)


# COMMAND ----------

# MAGIC %md
# MAGIC ### D - RFM (Recency, Frequency, Monetary) – fact_wager

# COMMAND ----------

def get_features_d(analysis: str):

    # ------------------------------
    # Overall RFM (combined scope)
    # ------------------------------
    rfm_all = (
        wager
        .filter(ANALYSIS_FILTER)
        .filter(F.to_date(wager_time_col) >= recent_start)
        .groupBy("user_id", "source_system")
        .agg(
            F.max(wager_time_col).alias("last_bet_time_90d"),
            F.countDistinct(F.to_date(wager_time_col)).alias("bet_days_90d"),
            F.sum(F.col(wager_stake_col)).alias("stake_sum_90d")
        )
        .withColumn("R_days", F.datediff(current_date, F.to_date("last_bet_time_90d")))
    )

    # Normalize 0–1 (overall)
    stats = rfm_all.agg(
        F.min("R_days").alias("R_min"), F.max("R_days").alias("R_max"),
        F.min("bet_days_90d").alias("F_min"), F.max("bet_days_90d").alias("F_max"),
        F.min("stake_sum_90d").alias("M_min"), F.max("stake_sum_90d").alias("M_max")
    ).collect()[0]

    R_min, R_max = stats["R_min"], stats["R_max"]
    F_min, F_max = stats["F_min"], stats["F_max"]
    M_min, M_max = stats["M_min"], stats["M_max"]

    eps = 1e-9

    features_all = (
        rfm_all
        .withColumn("R_norm", (F.lit(R_max) - F.col("R_days")) / (F.lit(R_max) - F.lit(R_min) + eps))
        .withColumn("F_norm", (F.col("bet_days_90d") - F.lit(F_min)) / (F.lit(F_max) - F.lit(F_min) + eps))
        .withColumn("M_norm", (F.col("stake_sum_90d") - F.lit(M_min)) / (F.lit(M_max) - F.lit(M_min) + eps))
        .withColumn("rfm_score", 0.4 * F.col("R_norm") + 0.3 * F.col("F_norm") + 0.3 * F.col("M_norm"))
    )

    # ------------------------------
    # Product-specific raw RFM inputs (no normalization to avoid mixing distributions)
    # ------------------------------
    def _rfm_raw(filter_expr, suffix: str):
        return (
            wager
            .filter(filter_expr)
            .filter(F.to_date(wager_time_col) >= recent_start)
            .groupBy("user_id", "source_system")
            .agg(
                F.max(wager_time_col).alias(f"last_bet_time_90d{suffix}"),
                F.countDistinct(F.to_date(wager_time_col)).alias(f"bet_days_90d{suffix}"),
                F.sum(F.col(wager_stake_col)).alias(f"stake_sum_90d{suffix}")
            )
            .withColumn(f"R_days{suffix}", F.datediff(current_date, F.to_date(F.col(f"last_bet_time_90d{suffix}"))))
        )

    rfm_online = _rfm_raw(ONLINE_FILTER, "_online")
    rfm_live   = _rfm_raw(LIVE_FILTER, "_live")

    features_D = (
        features_all
        .join(rfm_online, on=["user_id","source_system"], how="left")
        .join(rfm_live, on=["user_id","source_system"], how="left")
        .fillna({
            "bet_days_90d_online": 0,
            "stake_sum_90d_online": 0.0,
            "bet_days_90d_live": 0,
            "stake_sum_90d_live": 0.0,
        })
        # Cross-product preference / mix (within 90d)
        .withColumn(
            "live_share_of_stake_90d",
            F.when(F.col("stake_sum_90d") > 0, F.col("stake_sum_90d_live") / F.col("stake_sum_90d")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "live_share_of_bet_days_90d",
            F.when(F.col("bet_days_90d") > 0, F.col("bet_days_90d_live") / F.col("bet_days_90d")).otherwise(F.lit(0.0))
        )
    )

    return features_D

# features_D = get_features_d("analysis_5")
# display(features_D)


# COMMAND ----------

# MAGIC %md
# MAGIC ### E - Context / Profile (user_profile + wallet.coupon)

# COMMAND ----------

def get_features_e(analysis: str):

    coupon_features = (
        wager
        .filter(ANALYSIS_FILTER)
        .groupBy("user_id","source_system")
        .agg(
            F.countDistinct("coupon_id").alias("num_coupons"),
            F.max(coupon_valid_end_date_col).alias("last_coupon_time")
        )
    )

    features_E = coupon_features.fillna({"num_coupons": 0})

    return features_E

# features_E = get_features_e("analysis_5")
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
        .join(features_B, on=["user_id","source_system"], how="left")
        .join(features_C, on=["user_id","source_system"], how="left")
        .join(features_D, on=["user_id","source_system"], how="left")
        .join(features_E, on=["user_id","source_system"], how="left")
        .dropDuplicates(["user_id","source_system"])
    )

    return features

# features = get_all_features("analysis_5")
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