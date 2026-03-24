# Databricks notebook source
# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# Shared MBA configuration
MBA_ANALYSIS = "analysis_4"  # Online Casino + Live Casino + Sports
MBA_LOOKBACK_DAYS_LIST = [30, 365]

# Feature engineering thresholds
MBA_MIN_BETS_PER_ITEM = 3
MBA_MIN_WAGER_PER_ITEM = 50.0
MBA_MIN_ITEMS_PER_USER = 2

# Rule mining thresholds
MBA_MIN_SUPPORT = 0.003
MBA_MIN_CONFIDENCE = 0.10
MBA_MIN_LIFT = 1.05
MBA_TOP_RULES = 500

# Fallback (when Spark FPGrowth is unsupported in Spark Connect)
MBA_FALLBACK_MIN_SUPPORT = 0.01
MBA_FALLBACK_MIN_CONFIDENCE = 0.15
MBA_FALLBACK_TOP_ITEMS = 400
MBA_FALLBACK_MAX_USERS = 250000
MBA_FALLBACK_SAMPLE_SEED = 42
MBA_PAIRWISE_MAX_ITEMS_PER_USER = 80

# MBA tables
T_MBA_USER_ITEM_FEATURES_PREFIX = f"{WRITE_SCHEMA}.mba_user_item_features"
T_MBA_TRANSACTIONS_PREFIX = f"{WRITE_SCHEMA}.mba_transactions"
T_MBA_RULES_PREFIX = f"{WRITE_SCHEMA}.mba_association_rules"
T_MBA_RULES_CROSS_PREFIX = f"{WRITE_SCHEMA}.mba_association_rules_cross"
T_MBA_RULES_FOCUS_PREFIX = f"{WRITE_SCHEMA}.mba_association_rules_focus"
