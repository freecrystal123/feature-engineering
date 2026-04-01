# Databricks notebook source
# MAGIC %md
# MAGIC ## 5.3 - MBA Features Debug (Horse Racing / Unknown Sport)
# MAGIC
# MAGIC Goal:
# MAGIC - Rebuild, in SQL, the same ETL logic from `050_mba_features`.
# MAGIC - Trace two target users step-by-step.
# MAGIC - Explain where `sports::Unknown Sport` is created.
# MAGIC - Compare reconstructed outputs with official MBA tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0) Inputs and constants
# MAGIC
# MAGIC Update this first cell if you need to debug more users.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_target_users AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('PLAY971-2512061611360000115'),
# MAGIC   ('PLAY971-2512061621510000122')
# MAGIC AS t(unique_user_id);
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_lookbacks AS
# MAGIC SELECT * FROM VALUES
# MAGIC   (30),
# MAGIC   (365)
# MAGIC AS t(lookback_days);
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_target_user_keys AS
# MAGIC SELECT
# MAGIC   unique_user_id,
# MAGIC   split(unique_user_id, '-')[0] AS source_system,
# MAGIC   regexp_extract(unique_user_id, '^[^-]+-(.*)$', 1) AS user_id
# MAGIC FROM dbg_target_users;
# MAGIC
# MAGIC -- Keep thresholds aligned with cross_sell_mba/00_config.py
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_thresholds AS
# MAGIC SELECT
# MAGIC   3   AS min_bets_per_item,
# MAGIC   50.0 AS min_wager_per_item,
# MAGIC   2   AS min_items_per_user;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Quick check on official output tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM uat_analytics.cross_sell_mba.mba_user_item_features_ols
# MAGIC WHERE unique_user_id IN (SELECT unique_user_id FROM dbg_target_users)
# MAGIC ORDER BY unique_user_id, lookback_days, vertical, item_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM uat_analytics.cross_sell_mba.mba_transactions_ols
# MAGIC WHERE unique_user_id IN (SELECT unique_user_id FROM dbg_target_users)
# MAGIC ORDER BY unique_user_id, lookback_days;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Build lookup views used by the ETL

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check column names first (some environments may use sport_type vs sport_tyoe)
# MAGIC DESCRIBE TABLE prod_silver.wager.wager_history;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE prod_creative.shared.dim_sport_type;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_sports_lookup AS
# MAGIC SELECT
# MAGIC   CAST(w.id AS STRING) AS wh_wager_id,
# MAGIC   CAST(w.source_system AS STRING) AS wh_source_system,
# MAGIC   MAX(CAST(w.sport_type AS INT)) AS sport_type
# MAGIC FROM prod_silver.wager.wager_history w
# MAGIC WHERE w.id IS NOT NULL
# MAGIC   AND w.source_system IS NOT NULL
# MAGIC GROUP BY
# MAGIC   CAST(w.id AS STRING),
# MAGIC   CAST(w.source_system AS STRING);
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_sports_type_lookup AS
# MAGIC SELECT DISTINCT
# MAGIC   CAST(s.sport_tyoe AS INT) AS sport_type,
# MAGIC   CAST(s.sport_name AS STRING) AS sport_name
# MAGIC FROM prod_creative.shared.dim_sport_type s;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) Rebuild base events exactly like 050_mba_features
# MAGIC
# MAGIC Notes:
# MAGIC - Scope: categories (1, 3, 6), non-test users, non-lottery.
# MAGIC - `lookback_days` is exploded with 30 and 365 windows.
# MAGIC - We keep diagnostic flags for each join to explain Unknown Sport.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_base_events AS
# MAGIC SELECT
# MAGIC   lb.lookback_days,
# MAGIC   tu.unique_user_id,
# MAGIC
# MAGIC   CAST(u.user_id AS STRING) AS user_id,
# MAGIC   CAST(f.source_system AS STRING) AS source_system,
# MAGIC
# MAGIC   f.user_sk,
# MAGIC   f.game_sk,
# MAGIC   f.wager_sk,
# MAGIC   CAST(dw.dw_wager_id AS STRING) AS dw_wager_id,
# MAGIC   CAST(dw.dw_source_system AS STRING) AS dw_source_system,
# MAGIC
# MAGIC   TO_DATE(f.wager_settlement_timestamp) AS event_date,
# MAGIC   CAST(COALESCE(f.wager_amount, 0.0) AS DOUBLE) AS wager_amount,
# MAGIC
# MAGIC   g.game_category_id,
# MAGIC   g.game_name,
# MAGIC   wh.sport_type AS wh_sport_type,
# MAGIC   dst.sport_name AS dst_sport_name,
# MAGIC
# MAGIC   CASE WHEN g.game_category_id = 6 THEN 'sports' ELSE 'casino' END AS vertical,
# MAGIC   CASE
# MAGIC     WHEN g.game_category_id = 6 THEN COALESCE(dst.sport_name, 'Unknown Sport')
# MAGIC     ELSE COALESCE(g.game_name, 'Unknown Casino Game')
# MAGIC   END AS item_name,
# MAGIC
# MAGIC   CASE WHEN dw.dw_wager_sk IS NULL THEN 1 ELSE 0 END AS flag_missing_dim_wager,
# MAGIC   CASE WHEN wh.wh_wager_id IS NULL THEN 1 ELSE 0 END AS flag_missing_wager_history,
# MAGIC   CASE
# MAGIC     WHEN g.game_category_id = 6 AND dst.sport_name IS NULL THEN 1
# MAGIC     ELSE 0
# MAGIC   END AS flag_unknown_sport
# MAGIC
# MAGIC FROM prod_gold.finance.fact_settled_wager f
# MAGIC INNER JOIN prod_gold.shared.dim_game g
# MAGIC   ON f.game_sk = g.game_sk
# MAGIC  AND f.source_system = g.source_system
# MAGIC INNER JOIN prod_gold.shared.dim_user u
# MAGIC   ON f.user_sk = u.user_sk
# MAGIC  AND f.source_system = u.source_system
# MAGIC INNER JOIN dbg_target_user_keys tu
# MAGIC   ON CAST(u.user_id AS STRING) = tu.user_id
# MAGIC  AND CAST(f.source_system AS STRING) = tu.source_system
# MAGIC CROSS JOIN dbg_lookbacks lb
# MAGIC LEFT JOIN (
# MAGIC   SELECT
# MAGIC     wager_sk AS dw_wager_sk,
# MAGIC     CAST(wager_id AS STRING) AS dw_wager_id,
# MAGIC     CAST(source_system AS STRING) AS dw_source_system
# MAGIC   FROM prod_gold.shared.dim_wager
# MAGIC ) dw
# MAGIC   ON f.wager_sk = dw.dw_wager_sk
# MAGIC  AND CAST(f.source_system AS STRING) = dw.dw_source_system
# MAGIC LEFT JOIN dbg_sports_lookup wh
# MAGIC   ON dw.dw_wager_id = wh.wh_wager_id
# MAGIC  AND dw.dw_source_system = wh.wh_source_system
# MAGIC LEFT JOIN dbg_sports_type_lookup dst
# MAGIC   ON wh.sport_type = dst.sport_type
# MAGIC WHERE u.is_test_user = FALSE
# MAGIC   AND UPPER(CAST(f.source_system AS STRING)) <> 'LOTTERY'
# MAGIC   AND g.game_category_id IN (1, 3, 6)
# MAGIC   AND TO_DATE(f.wager_settlement_timestamp) >= DATE_SUB(CURRENT_DATE(), lb.lookback_days);

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) Event-level diagnostics (where Unknown Sport comes from)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   lookback_days,
# MAGIC   unique_user_id,
# MAGIC   COUNT(*) AS total_events,
# MAGIC   SUM(CASE WHEN vertical = 'sports' THEN 1 ELSE 0 END) AS sports_events,
# MAGIC   SUM(flag_unknown_sport) AS unknown_sport_events,
# MAGIC   SUM(flag_missing_dim_wager) AS missing_dim_wager_events,
# MAGIC   SUM(flag_missing_wager_history) AS missing_wager_history_events
# MAGIC FROM dbg_base_events
# MAGIC GROUP BY lookback_days, unique_user_id
# MAGIC ORDER BY lookback_days, unique_user_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_unknown_sport_events AS
# MAGIC SELECT *
# MAGIC FROM dbg_base_events
# MAGIC WHERE vertical = 'sports'
# MAGIC   AND flag_unknown_sport = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   lookback_days,
# MAGIC   unique_user_id,
# MAGIC   event_date,
# MAGIC   source_system,
# MAGIC   user_sk,
# MAGIC   wager_sk,
# MAGIC   dw_wager_id,
# MAGIC   dw_source_system,
# MAGIC   game_sk,
# MAGIC   game_name,
# MAGIC   game_category_id,
# MAGIC   wh_sport_type,
# MAGIC   dst_sport_name,
# MAGIC   flag_missing_dim_wager,
# MAGIC   flag_missing_wager_history,
# MAGIC   flag_unknown_sport
# MAGIC FROM dbg_unknown_sport_events
# MAGIC ORDER BY lookback_days, unique_user_id, event_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Drill to raw wager_history rows for the unknown cases
# MAGIC SELECT
# MAGIC   e.lookback_days,
# MAGIC   e.unique_user_id,
# MAGIC   e.event_date,
# MAGIC   e.dw_wager_id,
# MAGIC   e.dw_source_system,
# MAGIC   w.id AS wh_id,
# MAGIC   w.source_system AS wh_source_system,
# MAGIC   w.sport_type AS wh_sport_type
# MAGIC FROM dbg_unknown_sport_events e
# MAGIC LEFT JOIN prod_silver.wager.wager_history w
# MAGIC   ON CAST(w.id AS STRING) = e.dw_wager_id
# MAGIC  AND CAST(w.source_system AS STRING) = e.dw_source_system
# MAGIC ORDER BY e.lookback_days, e.unique_user_id, e.event_date;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5) Rebuild user_item_features (same logic as ETL)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_all_events AS
# MAGIC SELECT
# MAGIC   lookback_days,
# MAGIC   unique_user_id,
# MAGIC   source_system,
# MAGIC   user_id,
# MAGIC   vertical,
# MAGIC   item_name,
# MAGIC   TRIM(REGEXP_REPLACE(item_name, '\\s+', ' ')) AS item_name_clean,
# MAGIC   wager_amount,
# MAGIC   event_date
# MAGIC FROM dbg_base_events;
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_user_item_features AS
# MAGIC SELECT
# MAGIC   e.unique_user_id,
# MAGIC   e.source_system,
# MAGIC   e.user_id,
# MAGIC   e.vertical,
# MAGIC   CONCAT(e.vertical, '::', CASE WHEN e.item_name_clean = '' THEN 'Unknown' ELSE e.item_name_clean END) AS item_id,
# MAGIC   CASE WHEN e.item_name_clean = '' THEN 'Unknown' ELSE e.item_name_clean END AS item_name_clean,
# MAGIC   COUNT(*) AS num_bets,
# MAGIC   SUM(COALESCE(e.wager_amount, 0.0)) AS wager_amount_sum,
# MAGIC   MAX(e.event_date) AS last_event_date,
# MAGIC   CASE
# MAGIC     WHEN COUNT(*) >= (SELECT min_bets_per_item FROM dbg_thresholds)
# MAGIC       OR SUM(COALESCE(e.wager_amount, 0.0)) >= (SELECT min_wager_per_item FROM dbg_thresholds)
# MAGIC     THEN 1 ELSE 0
# MAGIC   END AS is_item_eligible,
# MAGIC   e.lookback_days
# MAGIC FROM dbg_all_events e
# MAGIC GROUP BY
# MAGIC   e.unique_user_id, e.source_system, e.user_id, e.vertical,
# MAGIC   CONCAT(e.vertical, '::', CASE WHEN e.item_name_clean = '' THEN 'Unknown' ELSE e.item_name_clean END),
# MAGIC   CASE WHEN e.item_name_clean = '' THEN 'Unknown' ELSE e.item_name_clean END,
# MAGIC   e.lookback_days;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM dbg_user_item_features
# MAGIC ORDER BY unique_user_id, lookback_days, vertical, item_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6) Rebuild transactions and compare with official table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW dbg_transactions AS
# MAGIC WITH eligible AS (
# MAGIC   SELECT *
# MAGIC   FROM dbg_user_item_features
# MAGIC   WHERE is_item_eligible = 1
# MAGIC ),
# MAGIC agg AS (
# MAGIC   SELECT
# MAGIC     unique_user_id,
# MAGIC     lookback_days,
# MAGIC     COLLECT_SET(item_id) AS items,
# MAGIC     SUM(CASE WHEN vertical = 'casino' THEN 1 ELSE 0 END) AS casino_items_count,
# MAGIC     SUM(CASE WHEN vertical = 'sports' THEN 1 ELSE 0 END) AS sports_items_count
# MAGIC   FROM eligible
# MAGIC   GROUP BY unique_user_id, lookback_days
# MAGIC )
# MAGIC SELECT
# MAGIC   unique_user_id,
# MAGIC   lookback_days,
# MAGIC   items,
# MAGIC   casino_items_count,
# MAGIC   sports_items_count,
# MAGIC   SIZE(items) AS num_items,
# MAGIC   CASE WHEN casino_items_count > 0 AND sports_items_count > 0 THEN 1 ELSE 0 END AS is_cross_platform_user
# MAGIC FROM agg
# MAGIC WHERE SIZE(items) >= (SELECT min_items_per_user FROM dbg_thresholds);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM dbg_transactions
# MAGIC ORDER BY unique_user_id, lookback_days;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Side-by-side comparison: reconstructed vs official
# MAGIC SELECT
# MAGIC   'reconstructed' AS origin,
# MAGIC   t.*
# MAGIC FROM dbg_transactions t
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'official' AS origin,
# MAGIC   o.unique_user_id,
# MAGIC   o.lookback_days,
# MAGIC   o.items,
# MAGIC   o.casino_items_count,
# MAGIC   o.sports_items_count,
# MAGIC   o.num_items,
# MAGIC   o.is_cross_platform_user
# MAGIC FROM uat_analytics.cross_sell_mba.mba_transactions_ols o
# MAGIC WHERE o.unique_user_id IN (SELECT unique_user_id FROM dbg_target_users)
# MAGIC ORDER BY unique_user_id, lookback_days, origin;
