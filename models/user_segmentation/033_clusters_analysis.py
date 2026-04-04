# Databricks notebook source
# MAGIC %run ./00_config_user_segmentation

# COMMAND ----------

# MAGIC %run ./00_llm_utils

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from uat_analytics.user_segmentation.cluster_online_live_casino_ap2

# COMMAND ----------

# DBTITLE 1,PCA renaming
# rename_pca("analysis_6", old_name="Outcome variability", new_name="Loss Dominance", new_description="Captures an outcome-skew axis driven by higher loss_rate vs lower win_rate, typically paired with lower net_profit_ratio and signals of loss-driven behavior (e.g., chasing/loss streaks). High scores indicate outcomes dominated by losses; low scores indicate outcomes dominated by wins.")
# rename_pca("analysis_6", old_name="Risk exposure", new_name="Loss Reactivity", new_description="Captures loss-response behavior and volatility, combining loss streak length, loss chasing (chasing_ratio), and stake variability (stake_stddev_ratio), often alongside time-since-activity signals. High scores indicate stronger reactive patterns after losses and more volatile stake sizing; low scores indicate more stable, less loss-reactive behavior.")
# rename_pca("analysis_6", old_name="Promotion Reliance", new_name="Promotional Engagement", new_description="Captures the degree to which engagement is driven by promotions, led by coupon usage (num_coupons) and contrasted with lower stake_to_deposit_ratio and lower monetary value (M_norm). High scores indicate promotion-driven activity; low scores indicate more organic play with less promotion involvement.")
# rename_pca("analysis_6", old_name="Monetary Pacing", new_name="Monetary Tempo", new_description="Captures the interaction between monetary scale (M_norm) and bet cadence (avg_time_between_bets), with secondary contrast from loss streaks and coupon usage. High scores indicate higher monetary value paired with slower, more spaced betting cadence; low scores indicate lower monetary value with faster cadence and more streak-driven patterns.")
rename_pca("analysis_6", old_name="Loss Reactivity", new_name="Post-Loss Disengagement", new_description="Captures a pattern where losing episodes (longer loss streaks and loss-chasing behavior) are associated with subsequent disengagement, reflected in higher time since last betting activity. High scores indicate stronger loss-linked withdrawal with more volatile stake sizing; low scores indicate steadier behavior with less post-loss drop-off.")
rename_pca("analysis_6", old_name="Loss Dominance", new_name="Loss-Weighted", new_description="Captures an outcome-skew axis where results are weighted toward losses (higher loss_rate and lower win_rate), often accompanied by lower net_profit_ratio and mild loss-driven signals (e.g., chasing). High scores indicate loss-leaning outcomes; low scores indicate win-leaning outcomes.")


# COMMAND ----------

# rename_cluster("analysis_6", old_name="Drifting risk takers", new_name="Post-Deposit Drift Risk Takers", new_description="Players showing a pronounced Deposit Without Activity pattern (recent deposits but less recent betting), combined with elevated Loss Dominance and Loss Reactivity signals (loss-heavy outcomes, longer loss streaks, and loss-chasing / stake variability). Overall, they exhibit risky, loss-responsive behavior alongside a clear misalignment between funding and wagering timelines.")
# rename_cluster("analysis_6", old_name="Passive Casuals", new_name="Promo-Engaged Light Regulars", new_description="Players with relatively regular presence but very low Session Intensity—they place few bets per session and show low Monetary Magnitude, while their engagement is strongly shaped by Promotional Engagement (coupon-driven activity). Overall, they participate lightly and consistently, with limited monetary involvement and minimal high-risk patterns.")

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np
from models.utils import get_model as get_registered_model

# COMMAND ----------

def get_model(analysis: str, version: int):
    analysis_name = ANALYSIS_NAMES[analysis]
    return get_registered_model(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
        version=version,
    )

# # pipeline esperado: PCA + KMeans
# pca = sk_model.named_steps["pca"]
# kmeans = sk_model.named_steps["kmeans"]

# centers = kmeans.cluster_centers_   # normalmente já está no espaço do PCA
print(get_model("analysis_6", 9))

# COMMAND ----------


import numpy as np
from pyspark.sql.types import NumericType, BooleanType

analysis = 'analysis_6'
version = 9

sk_model = get_model(analysis=analysis, version=version)

features = get_features_table(analysis).alias("f")

# --- Determine feature columns in the SAME order expected by the pipeline ---
model_feature_cols = None

# Prefer the first step that exposes feature_names_in_
if hasattr(sk_model, "feature_names_in_"):
    model_feature_cols = list(sk_model.feature_names_in_)
else:
    # try to find a step that has feature_names_in_
    for _, step in getattr(sk_model, "named_steps", {}).items():
        if hasattr(step, "feature_names_in_"):
            model_feature_cols = list(step.feature_names_in_)
            break

# fallback: numéricas/bool do schema (evita datas)
if model_feature_cols is None:
    model_feature_cols = [
        fld.name for fld in features.schema.fields
        if isinstance(fld.dataType, (NumericType, BooleanType))
        and fld.name not in {
            "unique_user_id", "user_id", "source_system", "cluster_id",
            "analysis", "analysis_id", "created_at", "updated_at"
        }
    ]
display(model_feature_cols)

# COMMAND ----------

get_predicted_table('analysis_6').select('cluster_id').distinct().display()

# COMMAND ----------

get_clusters_table('analysis_6').display()

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import pandas as pd

def get_cluster_extremes_and_boundaries(
    analysis: str,
    n_center: int = 5,
    n_boundary: int = 5,
    n_extreme: int = 5,
    version: int = 9,
):
    import numpy as np
    from pyspark.sql.types import NumericType, BooleanType

    sk_model = get_model(analysis=analysis, version=version)

    # --- Use the SAME preprocessing as training (everything up to, and including, PCA) ---
    # This assumes your pipeline ends with KMeans (named "kmeans") and has "pca" before it.
    # sk_model[:-1] will apply: (imputer/cleaning steps, scaler, pca, etc) exactly as trained.
    preproc = sk_model[:-1]

    # kmeans space == PCA space (after preprocessing)
    kmeans = sk_model.named_steps["kmeans"]
    centers = kmeans.cluster_centers_.astype("float64")
    K = centers.shape[0]

    # predicted: cluster_id = "cluster_1" etc
    predicted_raw = get_predicted_table(analysis).select("unique_user_id", "cluster_id").alias("p")

    # cria cluster_idx (int) a partir de "cluster_12" -> 11 (0-based)
    # predicted = (
    #     predicted_raw
    #     .withColumn("cluster_num", F.regexp_extract(F.col("cluster_id"), r"(\d+)", 1).cast("int"))
    #     .withColumn("cluster_idx", (F.col("cluster_num") - F.lit(1)).cast("int"))  # <- 0-based
    #     .drop("cluster_num")
    #     .alias("p")
    # )


    predicted = (
        predicted_raw
        .withColumn("cluster_idx", F.regexp_extract(F.col("cluster_id"), r"cluster_(\d+)", 1).cast("int"))
        .alias("p")
    )

    features = get_features_table(analysis).alias("f")
    clusters = get_clusters_table(analysis).alias("c")

    # --- Determine feature columns in the SAME order expected by the pipeline ---
    model_feature_cols = None

    # Prefer the first step that exposes feature_names_in_
    if hasattr(sk_model, "feature_names_in_"):
        model_feature_cols = list(sk_model.feature_names_in_)
    else:
        # try to find a step that has feature_names_in_
        for _, step in getattr(sk_model, "named_steps", {}).items():
            if hasattr(step, "feature_names_in_"):
                model_feature_cols = list(step.feature_names_in_)
                break

    # fallback: numéricas/bool do schema (evita datas)
    if model_feature_cols is None:
        model_feature_cols = [
            fld.name for fld in features.schema.fields
            if isinstance(fld.dataType, (NumericType, BooleanType))
            and fld.name not in {
                "unique_user_id", "user_id", "source_system", "cluster_id",
                "analysis", "analysis_id", "created_at", "updated_at"
            }
        ]

    # cast features para double na ordem do modelo
    select_exprs = [F.col(c).cast("double").alias(c) for c in model_feature_cols]

    base = (
        predicted
        .join(features.select("unique_user_id", *model_feature_cols), "unique_user_id", "inner")
        .select("unique_user_id", "cluster_id", "cluster_idx", *select_exprs)
        .where(F.col("cluster_idx").isNotNull())
    )

    centers_b = centers  # closure

    def compute_metrics(pdf: pd.DataFrame) -> pd.DataFrame:
        # NOTE:
        # - Do NOT do fillna(0) here blindly; let the pipeline handle missing values
        #   exactly as in training (imputer, etc.).
        # - If your pipeline does not include an imputer and cannot handle NaNs,
        #   you can optionally do pdf = pdf.fillna(0) BEFORE transform.
        X = pdf[model_feature_cols].to_numpy(dtype="float64")

        # Apply the FULL preprocessing pipeline (incl scaler + pca) exactly as training
        Z = preproc.transform(X)  # PCA-space representation

        diff = Z[:, None, :] - centers_b[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        own = pdf["cluster_idx"].astype("int64").to_numpy()

        invalid = (own < 0) | (own >= K)
        own_safe = own.copy()
        own_safe[invalid] = 0

        d1 = dists[np.arange(len(pdf)), own_safe]

        dists2 = dists.copy()
        dists2[np.arange(len(pdf)), own_safe] = np.inf
        d2 = dists2.min(axis=1)

        margin = d2 - d1

        # invalida métricas se cluster_idx inválido
        if invalid.any():
            d1 = d1.astype("float64"); d2 = d2.astype("float64"); margin = margin.astype("float64")
            d1[invalid] = np.nan
            d2[invalid] = np.nan
            margin[invalid] = np.nan

        out = pdf[["unique_user_id", "cluster_id", "cluster_idx"]].copy()
        out["dist_to_own_centroid"] = d1
        out["dist_to_2nd_centroid"] = d2
        out["margin_to_boundary"] = margin
        return out

    schema = StructType([
        StructField("unique_user_id", base.schema["unique_user_id"].dataType, False),
        StructField("cluster_id", base.schema["cluster_id"].dataType, False),
        StructField("cluster_idx", IntegerType(), False),
        StructField("dist_to_own_centroid", DoubleType(), True),
        StructField("dist_to_2nd_centroid", DoubleType(), True),
        StructField("margin_to_boundary", DoubleType(), True),
    ])

    metrics = (
        base
        .select("unique_user_id", "cluster_id", "cluster_idx", *model_feature_cols)
        .groupby("cluster_id")
        .applyInPandas(compute_metrics, schema=schema)
        .alias("m")
    )

    enriched = (
        metrics
        .join(features.select("unique_user_id", *model_feature_cols), "unique_user_id", "inner")
        .join(clusters, "cluster_id", "inner")
        .where(F.col("dist_to_own_centroid").isNotNull())
    )

    w_center   = Window.partitionBy("cluster_id").orderBy(F.col("dist_to_own_centroid").asc())
    w_boundary = Window.partitionBy("cluster_id").orderBy(F.col("margin_to_boundary").asc())
    w_extreme  = Window.partitionBy("cluster_id").orderBy(F.col("dist_to_own_centroid").desc())

    center_df = (
        enriched.withColumn("rn", F.row_number().over(w_center))
        .where(F.col("rn") <= n_center)
        .drop("rn")
        .withColumn("example_type", F.lit("near_centroid"))
    )

    boundary_df = (
        enriched.withColumn("rn", F.row_number().over(w_boundary))
        .where(F.col("rn") <= n_boundary)
        .drop("rn")
        .withColumn("example_type", F.lit("near_boundary"))
    )

    extreme_df = (
        enriched.withColumn("rn", F.row_number().over(w_extreme))
        .where(F.col("rn") <= n_extreme)
        .drop("rn")
        .withColumn("example_type", F.lit("near_upper_limit"))
    )

    result = (
        center_df
        .unionByName(boundary_df)
        .unionByName(extreme_df)
        .orderBy("cluster_id", "example_type", "dist_to_own_centroid")
    )

    return result

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

analysis_name = ANALYSIS_NAMES["analysis_6"]

analysis_column_1 = "days_since_last_bet"
analysis_column_2 = "days_since_last_deposit"

if analysis_column_2 == "":
    analysis_column_2 = analysis_column_1


base = (
    spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")
    .join(
        spark.table(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}"),
        on="unique_user_id",
        how="inner",
    )
    .join(
        spark.table(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}"),
        on="cluster_id",
        how="inner",
    )
    .withColumn(analysis_column_1, F.coalesce(F.col(analysis_column_1), F.lit(0)))
    .withColumn(analysis_column_2, F.coalesce(F.col(analysis_column_2), F.lit(0)))
)

df = (
    base
    .groupBy("cluster_id", "cluster_name", analysis_column_1, analysis_column_2)
    .agg(F.countDistinct("unique_user_id").alias("num_users"))
    .withColumn(
        "cluster_total_users",
        F.sum("num_users").over(Window.partitionBy("cluster_id"))
    )
    .withColumn(
        "pct_users",
        (F.col("num_users") / F.col("cluster_total_users"))
    )
    .select("cluster_id", "cluster_name", analysis_column_1, analysis_column_2, "num_users", "pct_users")  # ou mantenha num_users se quiser
    .orderBy("cluster_id", analysis_column_1)
)

display(df)

# COMMAND ----------

analysis_columns = ["R_days", "days_since_last_deposit"]


df = get_cluster_extremes_and_boundaries(
    analysis="analysis_6",
    n_center=5,
    n_boundary=5,
    n_extreme=5,
    version=9
)#.join(get_clusters_table('analysis_6').alias('c'),'cluster_id','inner').drop(F.col('c.cluster_id'), F.col("c.cluster_"))
display(df.select("cluster_id", "cluster_name_refined", *analysis_columns))
# display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- DORMANT DEPOSITORS
# MAGIC
# MAGIC -- select      sum(case when days_since_last_deposit < days_since_last_bet then 1 else 0 end), sum(1) from (
# MAGIC select      c.cluster_name_refined, 
# MAGIC             f.days_since_last_deposit,
# MAGIC             f.days_since_last_bet
# MAGIC from        uat_analytics.user_segmentation.pred_online_live_casino_ap2 p
# MAGIC inner join  uat_analytics.user_segmentation.cluster_online_live_casino_ap2 c on c.cluster_id = p.cluster_id
# MAGIC                                                                               and upper(c.cluster_name_refined) like '%DORMANT DEPOS%'
# MAGIC inner join  uat_analytics.user_segmentation.feat_online_live_casino_ap2 f on f.unique_user_id = p.unique_user_id
# MAGIC order by    days_since_last_deposit
# MAGIC -- )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- BURSTY INFREQUENT
# MAGIC
# MAGIC select      c.cluster_name_refined, 
# MAGIC             f.active_weeks_3m,
# MAGIC             f.bet_days_90d,
# MAGIC             f.avg_bets_per_day,
# MAGIC             f.interval_volatility
# MAGIC from        uat_analytics.user_segmentation.pred_online_live_casino_ap2 p
# MAGIC inner join  uat_analytics.user_segmentation.cluster_online_live_casino_ap2 c on c.cluster_id = p.cluster_id
# MAGIC                                                                               and upper(c.cluster_name_refined) like '%BURSTY%'
# MAGIC inner join  uat_analytics.user_segmentation.feat_online_live_casino_ap2 f on f.unique_user_id = p.unique_user_id
# MAGIC
# MAGIC -- order by    days_since_last_deposit

# COMMAND ----------

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F

analysis = "analysis_6"
version = 9

n_center   = 5
n_boundary = 50
n_extreme  = 5

TOP_N_PCS_FOR_PLOT = 2

marker_map = {
    "near_centroid": "s",
    "near_boundary": "^",
    "near_upper_limit": "o",
}

pred = get_predicted_table(analysis).select("*")

examples = get_cluster_extremes_and_boundaries(
    analysis=analysis,
    n_center=n_center,
    n_boundary=n_boundary,
    n_extreme=n_extreme,
    version=version
).select("unique_user_id", "cluster_id", "example_type")

clusters = get_clusters_table(analysis).select("*")

# PCA metadata (names)
pca_meta = get_pca_table(analysis).select("pca_id", "pca_component_name")
pca_name_map = {r["pca_id"]: r["pca_component_name"] for r in pca_meta.collect()}

# PCA columns from predicted table
pca_cols_pred = [c for c in pred.columns if re.match(r"^pca_\d+$", c)]
pca_cols_pred = sorted(pca_cols_pred, key=lambda s: int(s.split("_")[1]))
if len(pca_cols_pred) < 2:
    raise ValueError(f"Not enough PCA columns in predicted table. Found: {pca_cols_pred}")

# best name column
cluster_name_col = (
    "cluster_name_refined" if "cluster_name_refined" in clusters.columns else
    "cluster_name" if "cluster_name" in clusters.columns else
    None
)

clusters_named = (
    clusters.select(
        F.trim(F.col("cluster_id")).alias("cluster_id"),
        F.col(cluster_name_col).alias("cluster_name_display"),
        *[c for c in clusters.columns if c != "cluster_id"]
    )
    if cluster_name_col else
    clusters.select(
        F.trim(F.col("cluster_id")).alias("cluster_id"),
        F.col("cluster_id").alias("cluster_name_display"),
        *[c for c in clusters.columns if c != "cluster_id"]
    )
)

# choose top-2 PCs using intensity fields already in clusters table
strength_map = {
    "very_high": 2,
    "high": 1,
    "medium": 0,
    "low": 1,
    "very_low": 2,
    "neutral": -1,
    None: -1
}

cluster_rows = clusters_named.select(
    "cluster_id", "cluster_name_display",
    *[f"{c}_mean" for c in pca_cols_pred if f"{c}_mean" in clusters_named.columns],
    *[f"{c}_intensity" for c in pca_cols_pred if f"{c}_intensity" in clusters_named.columns],
).collect()

top2_by_cluster = {}
cluster_name_by_cluster = {}

for r in cluster_rows:
    rd = r.asDict(recursive=True)
    cid = str(rd["cluster_id"]).strip()
    cluster_name_by_cluster[cid] = rd.get("cluster_name_display", cid)

    items = []
    for p in pca_cols_pred:
        intensity_col = f"{p}_intensity"
        mean_col = f"{p}_mean"

        intensity = rd.get(intensity_col, None)
        strength = strength_map.get(intensity, -1)

        if strength >= 1:
            mean_val = float(rd.get(mean_col, 0.0) or 0.0)
            items.append((strength, abs(mean_val), p, intensity))

    items.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if len(items) >= TOP_N_PCS_FOR_PLOT:
        chosen = [items[i][2] for i in range(TOP_N_PCS_FOR_PLOT)]
    else:
        chosen = pca_cols_pred[:TOP_N_PCS_FOR_PLOT]

    top2_by_cluster[cid] = chosen

# --- FULL population (all users) for background (gray) ---
full_pred = (
    pred.select(
        "unique_user_id",
        F.trim(F.col("cluster_id")).alias("cluster_id"),
        *pca_cols_pred
    )
    .join(clusters_named.select("cluster_id", "cluster_name_display"), "cluster_id", "left")
)

pdf_full = full_pred.select("cluster_id", "cluster_name_display", *pca_cols_pred).toPandas()
pdf_full["cluster_id"] = pdf_full["cluster_id"].astype(str).str.strip()

# --- examples (selected points) ---
ex = (
    examples.select(
        "unique_user_id",
        F.trim(F.col("cluster_id")).alias("cluster_id"),
        F.trim(F.col("example_type")).alias("example_type"),
    ).alias("e")
    .join(
        pred.select(
            "unique_user_id",
            F.trim(F.col("cluster_id")).alias("cluster_id"),
            *pca_cols_pred
        ).alias("p"),
        ["unique_user_id", "cluster_id"],
        "inner"
    )
    .join(clusters_named.select("cluster_id", "cluster_name_display"), "cluster_id", "left")
)

pdf_all = ex.select("cluster_id", "cluster_name_display", "unique_user_id", "example_type", *pca_cols_pred).toPandas()
pdf_all["cluster_id"] = pdf_all["cluster_id"].astype(str).str.strip()
pdf_all["example_type"] = pdf_all["example_type"].astype(str).str.strip()

cluster_ids = sorted(
    pd.Series(pdf_full["cluster_id"].unique()).dropna().astype(str).str.strip().unique(),
    key=lambda s: int(re.search(r"(\d+)$", s).group(1)) if re.search(r"(\d+)$", s) else 0
)
n_clusters = len(cluster_ids)

# centroids (optional X)
sk_model = get_model(analysis=analysis, version=version)
kmeans = sk_model.named_steps["kmeans"]
centers = kmeans.cluster_centers_.astype("float64")

def pca_idx(p):
    m = re.match(r"^pca_(\d+)$", p)
    return int(m.group(1)) - 1 if m else None

ncols = 3 if n_clusters >= 3 else n_clusters
nrows = int(np.ceil(n_clusters / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
if n_clusters == 1:
    axes = np.array([axes])
axes = axes.flatten()

for i, cid in enumerate(cluster_ids):
    ax = axes[i]

    pcs = top2_by_cluster.get(cid, pca_cols_pred[:2])
    xcol, ycol = pcs[0], pcs[1]

    sub_full = pdf_full[pdf_full["cluster_id"] == cid]
    sub_ex   = pdf_all[pdf_all["cluster_id"] == cid].copy()

    # 1) Background: all users in the cluster (gray)
    if len(sub_full) > 0:
        ax.scatter(
            sub_full[xcol], sub_full[ycol],
            s=8, alpha=0.10, c="gray", label="all users"
        )

    # 2) Examples on top
    for etype, mk in marker_map.items():
        part = sub_ex[sub_ex["example_type"] == etype]
        if len(part) > 0:
            kwargs = {}
            if etype == "near_upper_limit":
                kwargs["c"] = "red"
            ax.scatter(part[xcol], part[ycol], marker=mk, label=etype, **kwargs)

    # 3) Centroid marker
    m = re.search(r"cluster_(\d+)$", cid)
    if m:
        cidx = int(m.group(1))
        ix = pca_idx(xcol)
        iy = pca_idx(ycol)
        if ix is not None and iy is not None and cidx < centers.shape[0] and ix < centers.shape[1] and iy < centers.shape[1]:
            ax.scatter(
                [centers[cidx, ix]], [centers[cidx, iy]],
                marker="X", s=220, c="green", label="centroid"
            )

    pca_x_name = pca_name_map.get(xcol, xcol)
    pca_y_name = pca_name_map.get(ycol, ycol)

    cname = (
        (sub_full["cluster_name_display"].iloc[0] if len(sub_full) else None)
        or (sub_ex["cluster_name_display"].iloc[0] if len(sub_ex) else None)
        or cluster_name_by_cluster.get(cid, cid)
    )

    ax.set_title(f"{cid} — {cname}\n({pca_x_name} vs {pca_y_name})")
    ax.set_xlabel(pca_x_name)
    ax.set_ylabel(pca_y_name)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.legend(loc="best", fontsize=9)

    # (optional) sanity print per cluster
    expected = n_center + n_boundary + n_extreme
    got = len(sub_ex)
    if got != expected:
        print(f"[WARN] {cid}: expected {expected}, got {got}. example_type counts:\n{sub_ex['example_type'].value_counts()}")

for j in range(n_clusters, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# audit
audit_rows = []
for cid in cluster_ids:
    pcs = top2_by_cluster.get(cid, pca_cols_pred[:2])
    audit_rows.append((
        cid,
        cluster_name_by_cluster.get(cid, cid),
        pcs[0], pca_name_map.get(pcs[0], pcs[0]),
        pcs[1], pca_name_map.get(pcs[1], pcs[1]),
    ))

display(
    spark.createDataFrame(
        audit_rows,
        ["cluster_id", "cluster_name", "pca_x", "pca_x_name", "pca_y", "pca_y_name"]
    )
)

# COMMAND ----------

get_pca_table('analysis_6').display()

# COMMAND ----------

# MAGIC %skip
# MAGIC def get_5_examples_from_each_cluster(analysis: str):
# MAGIC     analysis_name = ANALYSIS_NAMES[analysis]
# MAGIC     predicted = (
# MAGIC         get_predicted_table(analysis).groupby("cluster_id").applyInPandas(lambda pdf: pdf.head(5), schema=get_predicted_table(analysis).schema)
# MAGIC         .alias("p")
# MAGIC     )
# MAGIC     features = get_features_table(analysis)
# MAGIC     clusters = get_clusters_table(analysis).alias("c")
# MAGIC
# MAGIC     examples = (
# MAGIC         predicted
# MAGIC         .join(features, "unique_user_id", "inner")
# MAGIC         .join(clusters, F.col("p.cluster_id") == F.col("c.cluster_id"), "inner").drop(predicted.cluster_id)
# MAGIC         .select(F.col("c.cluster_id"),
# MAGIC                 F.col("c.cluster_name"),
# MAGIC                 F.col("c.cluster_name_refined"),
# MAGIC                 *[features[c] for c in features.columns])
# MAGIC         .orderBy(F.col("c.cluster_id"))
# MAGIC         # .display()
# MAGIC     )
# MAGIC
# MAGIC     return examples
# MAGIC get_5_examples_from_each_cluster('analysis_6')

# COMMAND ----------

# MAGIC %skip
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC def show_5_examples_grouped_by_cluster(
# MAGIC     analysis: str,
# MAGIC     limit_per_cluster: int = 5,
# MAGIC     order_by_cols=None,          # ex: ["-created_at"] ou ["-last_bet_time", "unique_user_id"]
# MAGIC     use_display: bool = True
# MAGIC ):
# MAGIC
# MAGIC
# MAGIC     df = get_5_examples_from_each_cluster(analysis)
# MAGIC
# MAGIC     cluster_id_col = "cluster_id"
# MAGIC     cluster_name_col = "cluster_name"
# MAGIC     cluster_name_refined_col = "cluster_name_refined"
# MAGIC
# MAGIC     id_cols = [cluster_id_col, cluster_name_col, cluster_name_refined_col]
# MAGIC     remaining_cols = [c for c in df.columns if c not in id_cols]
# MAGIC
# MAGIC     clusters = (
# MAGIC         df.select(*[F.col(c) for c in id_cols])
# MAGIC           .dropDuplicates()
# MAGIC           .orderBy(F.col(cluster_id_col))
# MAGIC           .collect()
# MAGIC     )
# MAGIC
# MAGIC     def _apply_order(dfx):
# MAGIC         if not order_by_cols:
# MAGIC             return dfx
# MAGIC         sort_exprs = []
# MAGIC         for c in order_by_cols:
# MAGIC             if isinstance(c, str) and c.startswith("-"):
# MAGIC                 sort_exprs.append(F.col(c[1:]).desc())
# MAGIC             else:
# MAGIC                 sort_exprs.append(F.col(c).asc())
# MAGIC         return dfx.orderBy(*sort_exprs)
# MAGIC
# MAGIC
# MAGIC     for r in clusters:
# MAGIC         cid = r[cluster_id_col]
# MAGIC         cname = r[cluster_name_col]
# MAGIC         cref = r[cluster_name_refined_col]
# MAGIC
# MAGIC         print(f"\n===  {cid}    {cname}  ->  {cref}  ===")
# MAGIC
# MAGIC         dfx = (
# MAGIC             df.filter(F.col(cluster_id_col) == cid)
# MAGIC               .select(*[F.col(c) for c in remaining_cols])
# MAGIC         )
# MAGIC
# MAGIC         dfx = _apply_order(dfx).limit(int(limit_per_cluster))
# MAGIC
# MAGIC         if use_display:
# MAGIC             display(dfx)
# MAGIC         else:
# MAGIC             dfx.show(truncate=False)
# MAGIC
# MAGIC
# MAGIC show_5_examples_grouped_by_cluster("analysis_6", limit_per_cluster=5, order_by_cols=["-created_at"])
