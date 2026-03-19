# Databricks notebook source
# MAGIC %md
# MAGIC # User Segmentation
# MAGIC ## 1 - feature engineering

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import NumericType
from pyspark.sql.functions import broadcast

# COMMAND ----------

# MAGIC %sql
# MAGIC select          fd.*
# MAGIC from            prod_gold.finance.fact_deposit fd
# MAGIC where           1=1
# MAGIC and             

# COMMAND ----------

# MAGIC %sql
# MAGIC select      distinct fw.source_system, dg.game_category_id, dg.game_category_description
# MAGIC from        prod_gold.finance.fact_wager fw
# MAGIC inner join  prod_gold.shared.dim_game dg on dg.game_sk = fw.game_sk
# MAGIC where       fw.date_sk >= 20250317

# COMMAND ----------

# MAGIC %sql
# MAGIC select          *
# MAGIC from            prod_gold.finance.fact_rtp

# COMMAND ----------

# MAGIC %sql
# MAGIC with 
# MAGIC user_amounts as (
# MAGIC     select          du.user_id,
# MAGIC                     sum(fsw.wager_amount) as wager_amount,
# MAGIC                     sum(zeroifnull(fd.vat_on_deposit_amount)) as vat_on_deposit_amount,
# MAGIC                     sum(zeroifnull(fr.rtp_amount)) as rtp_amount
# MAGIC     from            prod_gold.finance.fact_settled_wager fsw
# MAGIC     inner join      prod_gold.shared.dim_user du on du.user_sk = fsw.user_sk
# MAGIC                                                   and du.is_test_user = False
# MAGIC     inner join      prod_gold.shared.dim_game dg on dg.game_sk = fsw.game_sk
# MAGIC                                                   and dg.game_category_id in (1, 99)
# MAGIC     left join      prod_gold.finance.fact_deposit fd on fd.user_sk = fsw.user_sk
# MAGIC                                                     --   and fd.source_system = fsw.source_system
# MAGIC                                                       and fd.date_sk >= 20251020
# MAGIC                                                       and fd.date_sk <= 20260120
# MAGIC     left join      prod_gold.finance.fact_rtp fr on fr.user_sk = fsw.user_sk
# MAGIC                                                     --   and fr.source_system = fsw.source_system
# MAGIC                                                       and fr.date_sk >= 20251020
# MAGIC                                                       and fr.date_sk <= 20260120
# MAGIC     where           1=1
# MAGIC     and             fsw.date_sk >= 20251020
# MAGIC     and             fsw.date_sk <= 20260120
# MAGIC     group by    all
# MAGIC ),
# MAGIC
# MAGIC clusters as (
# MAGIC     select      *
# MAGIC     from        uat_analytics.user_segmentation.pred_online_casino
# MAGIC )
# MAGIC
# MAGIC select          c.cluster_id,
# MAGIC                 sum(zeroifnull(ua.wager_amount) - zeroifnull(ua.vat_on_deposit_amount) - zeroifnull(ua.rtp_amount)) as total_revenue
# MAGIC from            clusters c
# MAGIC left join      user_amounts ua on ua.user_id = c.user_id
# MAGIC group by all
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select      *
# MAGIC from        uat_analytics.user_segmentation.pred_online_casino
# MAGIC

# COMMAND ----------

# Parameters
GENERAL_DAYS_BEFORE = 365 # days
DAYS_RECENT = 90


recent_start = F.date_sub(F.current_date(), DAYS_RECENT)
first_start_analysis = F.date_sub(F.current_date(), GENERAL_DAYS_BEFORE)


# COMMAND ----------

# MAGIC %sql
# MAGIC select      count(distinct usr.user_id, usr.source_system) as user_quantity, fw.source_system
# MAGIC from        prod_gold.finance.fact_wager fw
# MAGIC inner join  prod_gold.shared.dim_user usr on fw.user_sk = usr.user_sk
# MAGIC                                           and is_test_user = false
# MAGIC where       fw.date_sk >= 20250101
# MAGIC group by    all

# COMMAND ----------

# MAGIC %sql
# MAGIC select      count(distinct wager_id) as wager_quantity, source_system
# MAGIC from        prod_gold.finance.fact_wager fw
# MAGIC where       fw.date_sk >= 20250101
# MAGIC group by source_system

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploring data between platforms

# COMMAND ----------

import matplotlib.pyplot as plt

# "analysis_1": (F.col("main_source_system") == "LOTTERY"),                                               # Lottery
# "analysis_2": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(3, 99))),   # Live Casino
# "analysis_3": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(1, 99))),   # Online Casino
# "analysis_4": ((F.col("main_source_system") != "LOTTERY") & (F.col("game_category_id").isin(6, 99))),   # Sports

wager = (
    spark.table('prod_gold.finance.fact_wager').alias("wager")
    .filter(F.to_date('wager_placement_timestamp') >= first_start_analysis)
    .join(
        spark.table("prod_gold.shared.dim_platform").alias("platform"),
        F.col("platform.platform_sk") == F.col("wager.platform_sk"),
        "left"
    )
    .join(
        spark.table("prod_gold.shared.dim_game").alias("game"),
        F.col("wager.game_sk") == F.col("game.game_sk"),
        "left"
    )
    .join(
        spark.table("prod_gold.shared.dim_user").alias("user"),
        F.col("wager.user_sk") == F.col("user.user_sk")
    )
    .withColumn(
        "segment_approach_1",
        F.when(F.col("platform_name") == "UAE Lottery", "Lottery")
        .when(F.col("game_category_id").isin(3,0), "Live Casino")
        .when(F.col("game_category_id") == 1, "Online Casino")
        .when(F.col("game_category_id") == 6, "Sports")
    )
    .withColumn(
        "segment_approach_2",
        F.col("platform_name")
    )
)

# Coletar segmentos
segments_1 = (
    wager.select("segment_approach_1")
    .distinct()
    .toPandas()["segment_approach_1"]
    .dropna()
    .tolist()
)

segments_2 = (
    wager.select("segment_approach_2")
    .distinct()
    .toPandas()["segment_approach_2"]
    .dropna()
    .tolist()
)

# Same row quantity
n_rows = max(len(segments_1), len(segments_2))

fig, axes = plt.subplots(
    n_rows, 2,
    figsize=(16, 4 * n_rows),
    sharex=False,
    sharey=False
)

if n_rows == 1:
    axes = [axes]

# ========= Left column: approach 1 =========
for i, segment in enumerate(segments_1):
    df = (
        wager
        .filter(F.col("segment_approach_1") == segment)
        .groupBy("user_id")
        .agg(F.sum("wager_amount").alias("total_wager_amount"))
        .toPandas()
        .astype({"total_wager_amount": float})
    )

    axes[i][0].boxplot(df["total_wager_amount"], vert=True)
    axes[i][0].set_title(f"Approach 1 — {segment}")
    axes[i][0].set_ylabel("Total Wager Amount")
    axes[i][0].grid(True, linestyle="--", alpha=0.5)

# ========= Right column: approach 2 =========
for i, segment in enumerate(segments_2):
    df = (
        wager
        .filter(F.col("segment_approach_2") == segment)
        .groupBy("user_id")
        .agg(F.sum("wager_amount").alias("total_wager_amount"))
        .toPandas()
        .astype({"total_wager_amount": float})
    )

    axes[i][1].boxplot(df["total_wager_amount"], vert=True)
    axes[i][1].set_title(f"Approach 2 — {segment}")
    axes[i][1].grid(True, linestyle="--", alpha=0.5)

# Títulos das colunas
# axes[0][0].set_xlabel("Approach 1 (Game Logic)")
# axes[0][1].set_xlabel("Approach 2 (Platform Logic)")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Because there are very significant outliers, used median to analyse the metrics:

# COMMAND ----------


def get_segments_statistics(segment_approach: str):
    df_segment = (
        wager
        .groupBy(F.col("user_id"), F.col(segment_approach))
        .agg(
            F.sum('wager_amount').alias("total_wager_amount"),
            F.count('*').alias("count_wagers_per_user")
        )
        # .orderBy(F.col("total_wager_amount").desc())
        .groupBy(F.col(segment_approach))
        .agg(
            F.median("count_wagers_per_user").alias("median_wager_quantity_per_user"),
            F.median("total_wager_amount").alias("median_wager_stake_per_user"),
            F.sum("total_wager_amount").alias("total_wager_amount"),
            F.sum("count_wagers_per_user").alias("total_wagers"),
            F.count('*').alias("total_users")
        )
    )

    pandas_df = df_segment.toPandas()
    return pandas_df


# COMMAND ----------

# MAGIC %skip
# MAGIC %sql
# MAGIC select      distinct game_name
# MAGIC from        prod_gold.finance.fact_wager fw
# MAGIC inner join  prod_gold.shared.dim_game dg on dg.game_sk = fw.game_sk
# MAGIC                                           and upper(dg.game_category_description) like '%ONLINE CASINO%'
# MAGIC where     fw.wager_placement_timestamp >= '2025-01-01'

# COMMAND ----------

display(get_segments_statistics('segment_approach_1'))
display(get_segments_statistics('segment_approach_2'))

# COMMAND ----------

# MAGIC %skip
# MAGIC (
# MAGIC     wager.filter(F.col("user_id") == "2407292121010004887")
# MAGIC         .orderBy(F.col("wager_placement_timestamp").desc())
# MAGIC         .agg(
# MAGIC             F.sum('wager_amount').alias("total_wager_amount"),
# MAGIC         )
# MAGIC         .display()
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select      *
# MAGIC from        uat_analytics.user_segmentation.feat_online_casino

# COMMAND ----------

# MAGIC %sql
# MAGIC select      pca.pca_component_name,
# MAGIC             pca.pca_component_description,
# MAGIC             pca.*
# MAGIC from        uat_analytics.user_segmentation.pca_online_casino pca

# COMMAND ----------

# MAGIC %sql
# MAGIC select      cls.cluster_id,
# MAGIC             cls.pca_1_intensity,
# MAGIC             cls.pca_2_intensity,
# MAGIC             cls.pca_3_intensity,
# MAGIC             cls.cluster_name_refined,
# MAGIC             cls.cluster_description_refined
# MAGIC             
# MAGIC from        uat_analytics.user_segmentation.cluster_online_casino cls

# COMMAND ----------

import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
# pandas_df_approach_2 =  get_segments_statistics('segment_approach_2')

# Approach 1 ----------------------------------------

global_max = max(
    get_segments_statistics('segment_approach_1')["total_wager_amount"].max(),
    get_segments_statistics('segment_approach_2')["total_wager_amount"].max()
)

for i, approach in enumerate(['segment_approach_1', 'segment_approach_2']):
    pandas_df =  get_segments_statistics(approach)
    segments = pandas_df[approach].unique()

    colors = dict(zip(segments, plt.cm.tab10.colors[:len(segments)]))
    for segment in segments:
        df = pandas_df[pandas_df[approach] == segment]
        axes[i].scatter(
            df["median_wager_quantity_per_user"],
            df["median_wager_stake_per_user"],
            s=(df["total_wager_amount"].astype(float).values
            / float(global_max)#pandas_df["total_wager_amount"].astype(float).max()
            * 2000),
            alpha=0.7,
            label=segment,
            color=colors[segment]
        )

    # Dots labels
    for _, row in pandas_df.iterrows():
        axes[i].text(
            row["median_wager_quantity_per_user"] * 1.05,
            row["median_wager_stake_per_user"] * 1.05,
            row[approach],
        )

    # axes[i].set_xscale("log")
    # axes[i].set_yscale("log")

    # plt.xlabel("Median Wager Quantity per User (log scale)")
    # plt.xlabel("Median Wager Quantity per User")
    axes[i].set_xlabel("Median Wager Quantity per User")
    # plt.ylabel("Median Wager Stake per User (log scale)")
    # plt.ylabel("Median Wager Stake per User")
    axes[i].set_ylabel("Median Wager Stake per User")
    axes[i].set_title(f"({approach}) Behavioral Regimes Across Segments")

    axes[i].grid(True, which="both", linestyle="--", linewidth=0.5)

# axes[1].set_xscale("log")
# axes[1].set_yscale("log")

# plt.legend(title="Platform")
# axes[0].show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach 1 (Analysis per Game Category)
# MAGIC ##### Visually, it is possible to observe that:
# MAGIC - The categories are significally distinct each other
# MAGIC - Comparing the size (total wager amount), **lottery** and **sports** could be grouped, but the games can't be compared
# MAGIC - **Lottery**, however, should be analyzed separately due to the inherently different nature of its games, and also has a really **low frequency**
# MAGIC
# MAGIC ### Approach 2 (Analysis per Platform)
# MAGIC ##### Visually, it is possible to observe that:
# MAGIC - **Truewin** represents a distinct case on its own
# MAGIC - **Dream Island** constitutes another separate case, really far from the other platforms
# MAGIC - **Play 971** and Sebaq may be grouped within the same analytical framework
# MAGIC - **Lottery**, however, should be analyzed separately due to the inherently different nature of its games, and also has a really **low frequency**

# COMMAND ----------

pandas_df_approach_1 = get_segments_statistics('segment_approach_1')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

axes[0]
# plt.figure(figsize=(8, 8))
axes[0].pie(
    pandas_df_approach_1["total_users"],
    labels=pandas_df_approach_1["segment_approach_1"],
    autopct='%1.1f%%',
    startangle=140
)
axes[0].set_title("Total Users per Game Category")
axes[0].axis('equal')

axes[1].pie(
    pandas_df_approach_1["total_wagers"],
    labels=pandas_df_approach_1["segment_approach_1"],
    autopct='%1.1f%%',
    startangle=140
)
axes[1].set_title("Total Wagers per Game Category")
axes[1].axis('equal')

# COMMAND ----------

# MAGIC %md
# MAGIC # FIX
# MAGIC
# MAGIC - This shows a problem of analysing it all, because **Online Casino** dominates the other categories, in terms of the **number of examples**  FIXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# MAGIC
# MAGIC - **Online Casino** tends to be **richer**, causing **more precision**, because of the **quantity of examples**
# MAGIC
# MAGIC - **Sports** categories turns into “**low activity noise**”
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Hybrid model:
# MAGIC
# MAGIC ###- Lottery (standalone)
# MAGIC - Low frequency
# MAGIC - Low ticket size
# MAGIC - Episodic engagement
# MAGIC - Risk metrics ≠ iGaming
# MAGIC
# MAGIC ###- Online + Live Casino (together)
# MAGIC
# MAGIC Although different, they share:
# MAGIC - High intensity
# MAGIC - High financial exposure
# MAGIC - Continuous sessions
# MAGIC - Relevant chasing and streak behaviors
# MAGIC - Chasing and streaks analysis makes sense 
# MAGIC
# MAGIC The difference between them is one of pace, not nature.
# MAGIC
# MAGIC ####Clustering learns:
# MAGIC - Impulsivity
# MAGIC - Control
# MAGIC - Value
# MAGIC - Volatility
# MAGIC
# MAGIC Not the game type itself.
# MAGIC
# MAGIC ###- Sports (own model)
# MAGIC - Moderate frequency (36)
# MAGIC - Intermediate average ticket size
# MAGIC - Strong dependence on:
# MAGIC   - odds
# MAGIC   - timing
# MAGIC   - event-driven behavior
# MAGIC
# MAGIC ####Sports betting:
# MAGIC - has highly relevant recency and interval patterns
# MAGIC - chasing is episodic, not continuous
# MAGIC
# MAGIC Mixing it with casino dilutes the signal.
# MAGIC
# MAGIC Lottery users clearly operate in a standalone regime, with very low wager frequency and monetary exposure, and must be modeled separately. Sports, Online Casino, and Live Casino also exhibit clearly distinct behavioral patterns. While Live and Online Casino show similar monetary exposure, they differ significantly in betting frequency and pace, indicating differences in behavioral rhythm rather than scale alone.
# MAGIC
# MAGIC
# MAGIC ##And what about the systems?
# MAGIC They do not disappear.
# MAGIC They are used as:
# MAGIC - an auxiliary feature (one-hot or categorical)
# MAGIC - a post-cluster analysis dimension
# MAGIC - an operational interpretation tool
# MAGIC
# MAGIC Never as the primary axis of clustering.
# MAGIC
# MAGIC ### Fails, because clustering will divide Live and Online users.
# MAGIC ### We don't need this division, we already have it.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Best approach: 4 analysis
# MAGIC ####- Online Casino
# MAGIC - high frequency
# MAGIC - high monetary
# MAGIC ####- Live Casino
# MAGIC - low frequency 
# MAGIC - high monetary
# MAGIC ####- Sports
# MAGIC - medium monetary
# MAGIC - low frequency
# MAGIC ####- Lottery
# MAGIC - low frequency
# MAGIC - low monetary
# MAGIC
# MAGIC **Online** and **Live Casino** share the same features.
# MAGIC
# MAGIC The difference between them, is the treatment on each feature. This will help PCA give each feature the importance it should have.