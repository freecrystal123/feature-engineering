# Databricks notebook source
# MAGIC %run ./00_config

# COMMAND ----------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# COMMAND ----------

TOP_N_FEATURES = 6 # top variables for PCA components naming
TOP_N_PCA_COMP = 0 # all

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {REPORT_SPACE}")

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

# COMMAND ----------

def get_pca_basic_info(analysis:str):
    analysis_name = ANALYSIS_NAMES[analysis]
    pca_table = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"

    pca_df = spark.table(pca_table).drop("pca_component_name", "pca_component_description")

    return pca_df

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


# COMMAND ----------

def save_report(analysis: str, html: str, report_name: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    report_path = f"{REPORT_DIR}/{report_name}_{analysis_name}.html"

    print(f"Report saved into {report_path}")
    # 1) salvar HTML
    with open(report_path, "w") as f:
        f.write(html)

# COMMAND ----------

def get_pca_profile_df(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    pred_df = spark.table(f'{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}')

    pca_cols = [c for c in pred_df.columns if c.startswith("pca_")]

    pca_profile_df = (
      pred_df.groupBy("cluster_id")
        .agg(
          F.count("*").alias("num_users"),
          *[F.avg(c).alias(f"avg_{c}") for c in pca_cols]
        )
    )

    return pca_cols, pca_profile_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster - analysis

# COMMAND ----------

def analyze_cluster_statistics(analysis: str):

    pca_cols, pca_profile_df = get_pca_profile_df(analysis)

    centroids_df = pca_profile_df

    pca_cols = [c for c in centroids_df.columns if c.startswith("avg_pca_")] 

    a = centroids_df.alias("a")
    b = centroids_df.alias("b")

    dist_df = (
        a.crossJoin(b)
        .where(F.col("a.cluster_id") < F.col("b.cluster_id"))  # avoids duplicate (A,B) e (B,A)
        .select(
            F.col("a.cluster_id").alias("cluster_a"),
            F.col("b.cluster_id").alias("cluster_b"),
            F.sqrt(
                sum([(F.col(f"a.{c}") - F.col(f"b.{c}"))**2 for c in pca_cols])
            ).alias("euclid_dist")
        )
        .orderBy("euclid_dist")
    )


    dot = sum([F.col(f"a.{c}") * F.col(f"b.{c}") for c in pca_cols])
    na = F.sqrt(sum([F.col(f"a.{c}")**2 for c in pca_cols]))
    nb = F.sqrt(sum([F.col(f"b.{c}")**2 for c in pca_cols]))

    cosine_df = (
        a.crossJoin(b)
        .where(F.col("a.cluster_id") < F.col("b.cluster_id"))
        .select(
            F.col("a.cluster_id").alias("cluster_a"),
            F.col("b.cluster_id").alias("cluster_b"),
            (F.lit(1) - (dot / (na * nb))).alias("cosine_dist")  # 0 = equal
        )
        .orderBy("cosine_dist")
    )


    displayHTML("## Predicted:")
    display(pred_df)
    displayHTML("## Centroids:")
    display(pca_profile_df)
    displayHTML("## Euclidean Distance:")
    display(dist_df)
    displayHTML("## Cosin Distance:")
    display(cosine_df)

# analyze_cluster_statistics('analysis_1')

# COMMAND ----------



# COMMAND ----------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import functions as F
import math

def get_cluster_radial_profiles(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]

    pca_cols, pca_profile_df = get_pca_profile_df(analysis)

    # --- PCA columns (ordered) ---
    pca_cols = sorted(
        [c for c in pca_profile_df.columns if c.startswith("avg_pca_")],
        key=lambda x: int(x.split("_")[-1])
    )

    labels = [c.replace("avg_", "") for c in pca_cols]
    pca_metadata = get_pca_metadata(analysis)
    labels = [pca_metadata[l]["name"] for l in labels]

    clusters_data = (
        spark.table(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}")
        .select(
            "cluster_id",
            # F.col("cluster_name"),
            # F.col("cluster_description"),
            F.col("cluster_name_refined").alias("cluster_name"),
            F.col("cluster_description_refined").alias("cluster_description")
            )
    )

    # --- Complete base (to scale and get num_users) ---
    all_pdf = (
        pca_profile_df
        .join(clusters_data, "cluster_id", "left")
        .select("cluster_id", "cluster_name", "cluster_description", "num_users", *pca_cols)
        .toPandas()
    )

    # Index per cluster_id
    all_pdf = all_pdf.set_index("cluster_id")

    # --- Scale each PCA component to [0,1] (between clusters) ---
    # vals = all_pdf[pca_cols]
    # mins = vals.min(axis=0)
    # maxs = vals.max(axis=0)
    # den = (maxs - mins).replace(0, np.nan)
    # vals_norm = ((vals - mins) / den).fillna(0.0)


    # Build quantile grid once (0..100%)
    probs = [i / 100 for i in range(101)]
    rel_err = 0.001  # adjust if needed

    global_quantiles = {
        k: pred_df.approxQuantile(k, probs, rel_err)
        for k in pca_keys
    }

    def value_to_percentile(v, q):
        """
        v: cluster mean value
        q: list of 101 quantiles (0..100%)
        returns percentile in [0,1] with light interpolation
        """
        import numpy as np
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.5  # neutral fallback

        # handle edges
        if v <= q[0]:
            return 0.0
        if v >= q[-1]:
            return 1.0

        # find bucket
        idx = np.searchsorted(q, v, side="right") - 1
        idx = max(0, min(idx, 99))

        q0, q1 = q[idx], q[idx + 1]
        if q1 == q0:
            return idx / 100.0

        # interpolate within the bucket
        frac = (v - q0) / (q1 - q0)
        return (idx + frac) / 100.0

    # Convert cluster avg_pca_* into percentile scores
    vals = all_pdf[pca_cols]  # cluster means
    vals_norm = vals.copy()

    for avg_col, key in zip(pca_cols, pca_keys):
        q = global_quantiles[key]
        vals_norm[avg_col] = vals[avg_col].apply(lambda v: value_to_percentile(v, q))




    # =========================
    # ORDER CLUSTERS (NORMAL / DEFAULT)
    # =========================
    clusters = list(all_pdf.index)

    def cluster_sort_key(cid):
        try:
            return int(str(cid).split("_")[-1])
        except Exception:
            return str(cid)

    cluster_ids = sorted(clusters, key=cluster_sort_key)

    # --- Subplots (one radar per cluster) ---
    n_cols = 3
    n = len(cluster_ids)
    n_rows = math.ceil(n / n_cols)

    def safe_str(x, fallback=""):
        if x is None:
            return fallback
        if isinstance(x, float) and np.isnan(x):
            return fallback
        return str(x)

    titles = [
        f"{cid} - {safe_str(all_pdf.loc[cid, 'cluster_name'], 'Unnamed')} (n={int(all_pdf.loc[cid,'num_users'])})"
        for cid in cluster_ids
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=titles
    )

    # Space between subplot_title and radar
    fig.update_annotations(yshift=30)
    fig.update_annotations(font=dict(size=20))

    fig.update_layout(margin=dict(t=110, l=40, r=40, b=40))
    fig.update_polars(angularaxis=dict(tickfont=dict(size=10)))

    for i, cid in enumerate(cluster_ids):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1

        fig.add_trace(
            go.Scatterpolar(
                r=vals_norm.loc[cid].values.tolist(),
                theta=labels,
                fill="toself",
                name=str(cid),
                hovertemplate=f"<b>{cid}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=c
        )

    fig.update_layout(height=350 * n_rows)

    # Apply range 0–1 in all 
    for k in range(1, n + 1):
        fig.layout[f"polar{k}"].radialaxis.update(visible=True, range=[0, 1])
        
    # displayHTML("## Cluster Profiles:")
    fig.show()

# COMMAND ----------

def get_pca_importance(analysis: str):

    NEUTRAL_EPS = 1e-6   # "close of zero"

    # pca_weights = get_pca_weights(analysis=analysis)
    pca_basic_info = get_pca_basic_info(analysis=analysis)

    names_df = (
        pca_basic_info
            .transform(get_pca_weights_unpivoted, analysis)
            .transform(get_num_features_above_threshold)
            .transform(get_pca_ranked)
            .transform(get_top_features)
    )


    df = names_df
    if "abs_weight" not in df.columns:
        df = df.withColumn("abs_weight", F.abs(F.col("weight")))

    w = Window.partitionBy("pca_id")

    df_labeled = (
        df
        # percentile relative to abs_weight inside PCA (0..1)
        .withColumn("weight_pct", F.percent_rank().over(w.orderBy(F.col("abs_weight"))))

        .withColumn(
            "strength",
            F.when(F.col("abs_weight") <= F.lit(NEUTRAL_EPS), F.lit(None))
            .when(F.col("weight_pct") >= 0.70, "strong")      # top 30%
            .when(F.col("weight_pct") >= 0.30, "moderate")    # middle 40%
            .otherwise("weak")                                # bottom 30% (non-neutral)
        )

        # direction (signal)
        .withColumn(
            "direction",
            F.when(F.col("abs_weight") <= F.lit(NEUTRAL_EPS), F.lit(None))
            .when(F.col("weight") > 0, "positive")
            .otherwise("negative")
        )

        # final label
        .withColumn(
            "signed_importance",
            F.when(F.col("abs_weight") <= F.lit(NEUTRAL_EPS), "neutral")
            .otherwise(F.concat_ws(" ", F.col("strength"), F.col("direction")))
        )
        .select("pca_id","feature_name","signed_importance")
    )
    return df_labeled


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster - general information

# COMMAND ----------

def get_html_cluster_general_information(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    analysis_pca = analysis

    pca_cols, pca_profile_df = get_pca_profile_df(analysis=analysis)

    # ======================================================
    # PCA columns
    # ======================================================
    pca_cols = sorted(
        [c for c in pca_profile_df.columns if c.startswith("avg_pca_")],
        key=lambda x: int(x.split("_")[-1])
    )

    pca_keys = [c.replace("avg_", "") for c in pca_cols]

    pca_metadata = get_pca_metadata(analysis)
    pca_labels = [pca_metadata.get(k, {}).get("name", k) for k in pca_keys]

    # ======================================================
    # Cluster metadata
    # ======================================================
    clusters_data = (
        spark.table(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}")
        .select(
            "cluster_id",
            F.col("cluster_name_refined").alias("cluster_name"),
            F.col("cluster_description_refined").alias("cluster_description")
        )
    )

    # ======================================================
    # Base dataframe
    # ======================================================
    all_pdf = (
        pca_profile_df
        .join(clusters_data, "cluster_id", "left")
        .select("cluster_id", "cluster_name", "cluster_description", "num_users", *pca_cols)
        .toPandas()
        .set_index("cluster_id")
    )

    all_pdf["num_users"] = (
        pd.to_numeric(all_pdf["num_users"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ======================================================
    # Order clusters by size
    # ======================================================
    cluster_ids = (
        all_pdf["num_users"]
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # ======================================================
    # Normalize PCA (min–max per component)
    # ======================================================
    # vals = all_pdf.loc[cluster_ids, pca_cols]
    # mins = vals.min(axis=0)
    # maxs = vals.max(axis=0)
    # den = (maxs - mins).replace(0, np.nan)
    # vals_norm = ((vals - mins) / den).fillna(0.0)

    pred_df = spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")

    # Build quantile grid once (0..100%)
    probs = [i / 100 for i in range(101)]
    rel_err = 0.001  # adjust if needed
    

    global_quantiles = {
        k: pred_df.approxQuantile(k, probs, rel_err)
        for k in pca_keys
    }

    def value_to_percentile(v, q):
        """
        v: cluster mean value
        q: list of 101 quantiles (0..100%)
        returns percentile in [0,1] with light interpolation
        """
        import numpy as np
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.5  # neutral fallback

        # handle edges
        if v <= q[0]:
            return 0.0
        if v >= q[-1]:
            return 1.0

        # find bucket
        idx = np.searchsorted(q, v, side="right") - 1
        idx = max(0, min(idx, 99))

        q0, q1 = q[idx], q[idx + 1]
        if q1 == q0:
            return idx / 100.0

        # interpolate within the bucket
        frac = (v - q0) / (q1 - q0)
        return (idx + frac) / 100.0

    # Convert cluster avg_pca_* into percentile scores
    vals = all_pdf[pca_cols]  # cluster means
    vals_norm = vals.copy()

    for avg_col, key in zip(pca_cols, pca_keys):
        q = global_quantiles[key]
        vals_norm[avg_col] = vals[avg_col].apply(lambda v: value_to_percentile(v, q))
    



    # ======================================================
    # Databricks palette
    # ======================================================
    DBR_PALETTE = ["#FF3621", "#2E6F9E", "#6E3CBC", "#00A3A3", "#FF9E1B", "#1B3139"]

    def rgba(hex_color, alpha):
        h = hex_color.lstrip("#")
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # ======================================================
    # Radar plots
    # ======================================================
    n_cols = 3
    n = len(cluster_ids)
    n_rows = math.ceil(n / n_cols)

    titles = [
        f"{all_pdf.loc[cid,'cluster_name']} (n={all_pdf.loc[cid,'num_users']})"
        for cid in cluster_ids
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "polar"}] * n_cols for _ in range(n_rows)],
        subplot_titles=titles
    )

    fig.update_annotations(yshift=30, font=dict(size=20, color="#E6EDF3"))
    fig.update_layout(margin=dict(t=110, l=40, r=40, b=40))

    # --- Force Plotly dark look (keep DBR palette traces) ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",  # transparent so panel CSS shows through
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
    )

    fig.update_polars(
        bgcolor="rgba(0,0,0,0)",
        angularaxis=dict(
            tickfont=dict(size=10, color="#E6EDF3"),
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
        ),
        radialaxis=dict(
            range=[0, 1],
            visible=True,
            tickfont=dict(color="#9BA3AF"),
            gridcolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
        ),
    )

    for i, cid in enumerate(cluster_ids):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        base = DBR_PALETTE[i % len(DBR_PALETTE)]

        fig.add_trace(
            go.Scatterpolar(
                r=vals_norm.loc[cid].values.tolist(),
                theta=pca_labels,
                fill="toself",
                mode="lines",
                line=dict(color=base, width=2),
                fillcolor=rgba(base, 0.35),
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(height=350 * n_rows)

    radar_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False},
    )

    # ======================================================
    # PCA IMPORTANCE 
    # ======================================================
    pca_importance_df = get_pca_importance(analysis).toPandas()

    # signed_importance examples:
    # "strong positive", "moderate positive", "weak positive", "neutral",
    # "weak negative", "moderate negative", "strong negative"

    def _parse_signed_importance(s: str):
        """
        Returns: (strength, direction)
        strength: strong|moderate|weak|neutral
        direction: positive|negative|neutral
        """
        if s is None:
            return ("neutral", "neutral")

        t = str(s).strip().lower()
        if t == "neutral" or t == "":
            return ("neutral", "neutral")

        parts = t.split()
        strength = parts[0] if len(parts) >= 1 else "neutral"
        direction = parts[1] if len(parts) >= 2 else "neutral"
        if strength not in {"strong", "moderate", "weak", "neutral"}:
            strength = "neutral"
        if direction not in {"positive", "negative", "neutral"}:
            direction = "neutral"
        return (strength, direction)

    def _arrow(strength: str, direction: str) -> str:
        # Option 3: arrows + subtle color + keep text label
        if direction == "neutral" or strength == "neutral":
            return "→"
        if direction == "positive":
            return {"strong": "↑↑", "moderate": "↑", "weak": "↑"}.get(strength, "↑")
        if direction == "negative":
            return {"strong": "↓↓", "moderate": "↓", "weak": "↓"}.get(strength, "↓")
        return "→"

    def _impact_class(strength: str, direction: str) -> str:
        if direction == "neutral" or strength == "neutral":
            return "impact-neutral"
        if direction == "positive":
            return {"strong": "impact-strong-pos", "moderate": "impact-mod-pos", "weak": "impact-weak-pos"}.get(strength, "impact-mod-pos")
        if direction == "negative":
            return {"strong": "impact-strong-neg", "moderate": "impact-mod-neg", "weak": "impact-weak-neg"}.get(strength, "impact-mod-neg")
        return "impact-neutral"

    parsed = pca_importance_df["signed_importance"].apply(_parse_signed_importance)
    pca_importance_df["strength"] = parsed.apply(lambda x: x[0])
    pca_importance_df["direction"] = parsed.apply(lambda x: x[1])
    pca_importance_df["arrow"] = pca_importance_df.apply(lambda r: _arrow(r["strength"], r["direction"]), axis=1)
    pca_importance_df["impact_class"] = pca_importance_df.apply(lambda r: _impact_class(r["strength"], r["direction"]), axis=1)

    # Keep the existing strength ordering inside each PCA
    strength_rank = {"strong": 0, "moderate": 1, "weak": 2, "neutral": 3}
    pca_importance_df["strength_rank"] = pca_importance_df["strength"].map(strength_rank).fillna(99).astype(int)

    # group features per PCA ordered by strength_rank
    features_by_pca = {}
    for pca_id, g in (
        pca_importance_df
        .sort_values(["pca_id", "strength_rank", "feature_name"])
        .groupby("pca_id")
    ):
        items = []
        for _, row in g.iterrows():
            arrow = row["arrow"]
            cls = row["impact_class"]
            label = row["signed_importance"]
            # Arrow + colored chip + label
            items.append(
                f"""
                <li class="feat">
                <span class="impact {cls}" title="{label}">{arrow}</span>
                <span class="fname"><b>{row['feature_name']}</b></span>
                <span class="ilabel muted">— {label}</span>
                </li>
                """
            )
        features_by_pca[pca_id] = items

    # ======================================================
    # PCA table HTML (with features)
    # ======================================================
    pca_rows_html = ""
    for key, label in zip(pca_keys, pca_labels):
        meta = pca_metadata.get(key, {})
        desc = meta.get("description", meta.get("desc", ""))

        feats = features_by_pca.get(key, [])
        feats_html = "<ul class='featlist'>" + "".join(feats) + "</ul>" if feats else "<i class='muted'>No dominant features</i>"

        pca_rows_html += f"""
        <tr>
        <td class="mono">{label}</td>
        <td class="muted">{desc}</td>
        <td>{feats_html}</td>
        </tr>
        """

    # ======================================================
    # Cluster cards 
    # ======================================================
    cluster_cards = []
    for i, cid in enumerate(cluster_ids):
        base = DBR_PALETTE[i % len(DBR_PALETTE)]
        name = all_pdf.loc[cid, "cluster_name"]
        desc = all_pdf.loc[cid, "cluster_description"]
        n_users = all_pdf.loc[cid, "num_users"]

        cluster_cards.append(f"""
        <details class="card">
        <summary>
            <div class="card-title">
            <span class="dot" style="background:{base};"></span>
            <span class="cname">{name}</span>
            </div>
            <div class="chip">n={n_users:,}</div>
        </summary>
        <div class="card-body">
            <div class="muted">{desc}</div>
        </div>
        </details>
        """)

    clusters_html = "\n".join(cluster_cards)

    # ======================================================
    # FINAL HTML
    # ======================================================
    html = f"""
    <div class="report" data-theme="dark">
    <div class="header">
        <div>
        <div class="title">Customer Segmentation — {analysis_name}</div>
        <div class="subtitle">Auto-generated report from PCA component definitions + cluster metadata + cluster profiles.</div>
        </div>
        <div class="badges">
        <span class="badge">{len(cluster_ids)} clusters</span>
        <span class="badge">{len(pca_cols)} PCA components</span>
        </div>
    </div>

    <div class="grid">
        <div class="panel span2">
        <div class="panel-title">Cluster Profiles (Radar)</div>
        <div class="panel-subtitle">Clusters are ordered by <b>num_users (descending)</b>. Values are min-max normalized per component across clusters (0–1).</div>
        <div class="plot">{radar_html}</div>
        </div>

        <div class="panel span2">
        <div class="panel-title">PCA Components & Feature Importance</div>
        <div class="panel-subtitle">Human-friendly component names + descriptions and feature-level explainability (arrow = direction, strength = intensity; label kept for clarity).</div>
        <div class="table-wrap">
            <table class="pca-table">
            <thead>
                <tr>
                <th style="width: 18%;">Component</th>
                <th style="width: 42%;">Description</th>
                <th style="width: 40%;">Top features (ordered by importance)</th>
                </tr>
            </thead>
            <tbody>
                {pca_rows_html}
            </tbody>
            </table>
        </div>
        </div>

        <div class="panel span2">
        <div class="panel-title">Clusters</div>
        <div class="panel-subtitle">Ordered by <b>num_users (descending)</b>. Open each cluster to read its business description.</div>
        <div class="cards">{clusters_html}</div>
        </div>
    </div>
    </div>

    <style>
    :root {{
        --bg: #F9F7F4;
        --panel: #FFFFFF;
        --text: #1B3139;
        --muted: #5E6A71;
        --border: rgba(27,49,57,0.12);
        --shadow: rgba(0,0,0,0.06);
        --chip: rgba(27,49,57,0.06);
    }}

    .report[data-theme="dark"] {{
        --bg: #0E1117;
        --panel: #161B22;
        --text: #E6EDF3;
        --muted: #9BA3AF;
        --border: rgba(255,255,255,0.08);
        --shadow: rgba(0,0,0,0.45);
        --chip: rgba(255,255,255,0.08);
    }}

    .report, .report * {{ box-sizing: border-box; }}
    .report {{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        background: var(--bg);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
    }}

    .header {{
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }}
    .title {{ font-size: 22px; font-weight: 800; letter-spacing: 0.2px; line-height: 1.2; }}
    .subtitle {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}

    .badges {{ display:flex; gap: 8px; flex-wrap: wrap; align-items:center; }}
    .badge {{
        font-size: 12px;
        color: var(--muted);
        border: 1px solid var(--border);
        background: var(--chip);
        padding: 6px 10px;
        border-radius: 999px;
        white-space: nowrap;
    }}

    .grid {{
        display:grid;
        grid-template-columns: 2fr 1fr;
        gap: 14px;
    }}

    .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 10px 25px var(--shadow);
        padding: 14px;
        min-width: 0;
    }}
    .span2 {{ grid-column: 1 / span 2; }}

    .panel-title {{ font-weight: 800; font-size: 14px; }}
    .panel-subtitle {{ font-size: 12px; color: var(--muted); margin-top: 4px; margin-bottom: 10px; line-height: 1.35; }}

    .plot {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--panel);
    }}

    .table-wrap {{
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
    }}

    .pca-table {{ width: 100%; border-collapse: collapse; }}
    .pca-table thead th {{
        text-align:left;
        font-size: 12px;
        color: var(--muted);
        font-weight: 700;
        background: var(--chip);
        padding: 10px 10px;
        border-bottom: 1px solid var(--border);
        position: sticky;
        top: 0;
        z-index: 1;
    }}
    .pca-table tbody td {{
        font-size: 13px;
        padding: 10px 10px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
        line-height: 1.35;
    }}

    .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        white-space: nowrap;
    }}
    .muted {{ color: var(--muted); }}

    /* Feature list styling (Option 3) */
    .featlist {{
        margin: 0;
        padding-left: 0;
        list-style: none;
    }}
    .feat {{
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin: 4px 0;
        line-height: 1.25;
    }}
    .impact {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 900;
        width: 28px;
        text-align: center;
        border-radius: 8px;
        padding: 2px 6px;
        border: 1px solid var(--border);
        background: var(--chip);
        flex: 0 0 auto;
    }}
    .fname {{
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .ilabel {{
        flex: 0 0 auto;
        white-space: nowrap;
    }}

    /* Arrow color semantics (subtle) */
    .impact-strong-pos {{ color: #00D18F !important; }}
    .impact-mod-pos    {{ color: #5CE6B8 !important; }}
    .impact-weak-pos   {{ color: #9AF3D6 !important; }}

    .impact-neutral    {{ color: #9BA3AF !important; }}

    .impact-weak-neg   {{ color: #FFB3A8 !important; }}
    .impact-mod-neg    {{ color: #FF7B6B !important; }}
    .impact-strong-neg {{ color: #FF3621 !important; }}

    .cards {{ display: grid; gap: 10px; }}

    details.card {{
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--panel);
        overflow: hidden;
    }}
    details.card > summary {{
        cursor: pointer;
        list-style: none;
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding: 12px 12px;
        gap: 12px;
    }}
    details.card[open] > summary {{
        background: var(--chip);
        border-bottom: 1px solid var(--border);
    }}
    summary::-webkit-details-marker {{ display:none; }}

    .card-title {{ display:flex; align-items:baseline; gap: 8px; min-width: 0; }}
    .dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; flex: 0 0 auto; }}
    .cname {{
        color: var(--text);
        font-weight: 800;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 70vw;
    }}
    .chip {{
        font-size: 12px;
        color: var(--muted);
        border: 1px solid var(--border);
        background: var(--chip);
        padding: 5px 9px;
        border-radius: 999px;
        white-space: nowrap;
        flex: 0 0 auto;
    }}
    .card-body {{ padding: 12px; }}

    @media (max-width: 1100px) {{
        .grid {{ grid-template-columns: 1fr; }}
        .span2 {{ grid-column: 1 / span 1; }}
        .cname {{ max-width: 85vw; }}
        .ilabel {{ display:none; }} /* optional: hide label on small screens */
    }}

    /* =======================================================
        FIX: Databricks displayHTML CSS overrides table text to black
        ======================================================= */
    .pca-table,
    .pca-table * {{
        color: var(--text) !important;
    }}

    .pca-table thead th {{
        color: var(--muted) !important;
    }}

    .pca-table tbody td {{
        color: var(--text) !important;
    }}

    .pca-table tbody td.muted,
    .pca-table .muted {{
        color: var(--muted) !important;
    }}

    /* Keep impact colors (override the global force above) */
    .impact.impact-strong-pos {{ color: #00D18F !important; }}
    .impact.impact-mod-pos    {{ color: #5CE6B8 !important; }}
    .impact.impact-weak-pos   {{ color: #AAAAAA !important; }}
    .impact.impact-neutral    {{ color: #9BA3AF !important; }}
    .impact.impact-weak-neg   {{ color: #AAAAAA !important; }}
    .impact.impact-mod-neg    {{ color: #FF7B6B !important; }}
    .impact.impact-strong-neg {{ color: #FF3621 !important; }}

    /* =======================================================
        PAGE BACKGROUND (Databricks notebook canvas)
        ======================================================= */
    html, body {{
        background: #000 !important;
    }}

    #app,
    #app > div,
    .notebook-container,
    .notebook-content,
    div[data-testid="notebook-container"],
    .output,
    .output_area,
    .cell-output,
    .cell-output-area {{
        background: #000 !important;
    }}

    .displayedHTML,
    .displayedHTML > div {{
        background: #000 !important;
    }}
    </style>
    """

    return html

    # displayHTML("## Built page:")
    # displayHTML(html)

# COMMAND ----------

def get_clusters_report_general_information(analysis: str):
    html = get_html_cluster_general_information(analysis)
    displayHTML(html)
    save_report(analysis=analysis, html=html, report_name="general_information")



# COMMAND ----------

for analysis, analysis_name in ANALYSIS_NAMES.items():
    get_clusters_report_general_information(analysis=analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster - detailed information

# COMMAND ----------



def get_html_cluster_detailed_information(analysis: str):

    analysis_name = ANALYSIS_NAMES[analysis]
    ANALYSIS_FILTER = ANALYSIS_FILTERS[analysis]

    recent_start = F.date_sub(F.current_date(), DAYS_RECENT)
    first_start_analysis = F.date_sub(F.current_date(), GENERAL_DAYS_BEFORE)
    wager_time_col = "wager_placement_timestamp_uae"
    wager_stake_col = "wager_amount"

    dim_wager = spark.table(DIM_WAGER)
    deposit = spark.table(T_DEPOSIT)
    withdraw = spark.table(T_WITHDRAW)
    user = spark.table(T_USER)
    dim_coupon = spark.table(DIM_COUPON)
    dim_game = spark.table(DIM_GAME)
    settled_wager = spark.table(T_SETTLED_WAGER)
    rtp = spark.table(T_RTP)

    wager = spark.table(T_WAGER)
    # wager = spark.table(T_WAGER).limit(10)
    wager = (
        wager
            .filter(F.to_date(wager_time_col) >= first_start_analysis)
            # .withColumnRenamed("source_system", "main_source_system")
            .withColumnRenamed("coupon_id", "wager_coupon_id")
            .join(
                user,
                ((wager.user_sk == user.user_sk) &
                (wager.source_system == user.source_system)),
                "left"
            ).drop(user.user_sk, user.source_system)
            .join(
                dim_wager,
                ((wager.wager_sk == dim_wager.wager_sk) &
                (wager.source_system == dim_wager.source_system)),
                "left"
            ).drop(dim_wager.wager_sk, dim_wager.source_system)
            .join(
                dim_coupon,
                ((wager.coupon_sk == dim_coupon.coupon_sk) &
                (wager.source_system == dim_coupon.source_system)),
                "left"
            ).drop(dim_coupon.source_system, dim_coupon.coupon_sk)
            .join(
                dim_game,
                ((wager.game_sk == dim_game.game_sk) &
                (wager.source_system == dim_game.source_system)),
                "left"
            ).drop(dim_game.source_system, dim_game.game_sk)
    )

    fact_settled_wager = (settled_wager
                                .filter(F.to_date('wager_settlement_timestamp') >= first_start_analysis)
                                .join(
                                    dim_game,
                                    ((settled_wager.game_sk == dim_game.game_sk) &
                                    (settled_wager.source_system == dim_game.source_system)),
                                    "left"
                                ).drop(dim_game.source_system, dim_game.game_sk)
                                .filter(ANALYSIS_FILTER)
                                .select("wager_sk", "source_system")
    )


    rtp_df = (rtp
                .filter(F.to_date('rtp_timestamp') >= first_start_analysis)
                .join(
                    dim_game,
                    ((rtp.game_sk == dim_game.game_sk) &
                    (rtp.source_system == dim_game.source_system)),
                    "left"
                ).drop(dim_game.source_system, dim_game.game_sk)
                .filter(ANALYSIS_FILTER)
    )

    DAYS_ALL = GENERAL_DAYS_BEFORE

    NON_LOTTERY_FILTER = (F.upper(F.col("source_system")) != F.lit("LOTTERY"))
    CAT_ALLOCATION_FILTER = F.col("game_category_id").isin(GAME_CATS_ALLOCATION)
    CAT_ANALYSIS_FILTER = ANALYSIS_FILTER

    def _apply_start_filter(df, *_):
        # Keep your original fixed window for revenue metrics
        # return df.filter((F.col("date_sk") >= 20251021) & (F.col("date_sk") < 20260120))

        start = F.date_sub(F.current_date(), GENERAL_DAYS_BEFORE)
        d = F.to_date(F.col("date_sk").cast("string"), "yyyyMMdd")
        return df.filter(d >= start)


    def _filter_last_days(df, days, ts_col=None, date_sk_col="date_sk"):
        """
        Robust filter for "last N days".
        - If date_sk exists: interpret as yyyyMMdd int/string.
        - Else use ts_col (timestamp/date string) and F.to_date.
        """
        start = F.date_sub(F.current_date(), int(days))
        if date_sk_col in df.columns:
            d = F.to_date(F.col(date_sk_col).cast("string"), "yyyyMMdd")
            return df.filter(d >= start)
        if ts_col is None:
            raise ValueError("ts_col must be provided when date_sk is not available.")
        return df.filter(F.to_date(F.col(ts_col)) >= start)

    # ======================================================
    # Load cluster assignments + cluster metadata
    # ======================================================
    predicted_df = (
        spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")
        .select(
            F.col("unique_user_id").cast("string").alias("unique_user_id"),
            F.col("cluster_id")
        )
        .dropDuplicates(["unique_user_id"])
    )

    clusters_meta_df = (
        spark.table(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}")
        .select(
            "cluster_id",
            F.col("cluster_name_refined").alias("cluster_name"),
            F.col("cluster_description_refined").alias("cluster_description"),
        )
    )

    # ======================================================
    # % USERS WHO ALSO PLAY LIVE (cat=3), SPORTS (cat=6), BOTH
    # Window: last DAYS_ALL days (same as activity)
    # Uses SETTLED wagers to avoid counting un-settled noise
    # ======================================================
    LIVE_CAT = 3
    SPORTS_CAT = 6

    fsw_presence = (
        _filter_last_days(settled_wager, days=DAYS_ALL, ts_col="wager_settlement_timestamp").alias("f")
        .join(
            dim_game.select("game_sk", "source_system", "game_category_id").alias("g"),
            (F.col("f.game_sk") == F.col("g.game_sk")) & (F.col("f.source_system") == F.col("g.source_system")),
            "inner"
        )
        # IMPORTANT: drop duplicated join keys from dim_game side to avoid ambiguity later
        .drop(F.col("g.game_sk"), F.col("g.source_system"))
        .filter(F.upper(F.col("f.source_system")) != F.lit("LOTTERY"))
        .filter(F.col("g.game_category_id").isin([LIVE_CAT, SPORTS_CAT]))
        .join(
            user.select("user_sk", "source_system", "user_id", "is_test_user").alias("u"),
            (F.col("f.user_sk") == F.col("u.user_sk")) & (F.col("f.source_system") == F.col("u.source_system")),
            "inner"
        )
        # IMPORTANT: drop duplicated join keys from user side too
        .drop(F.col("u.user_sk"))
        .filter(F.col("u.is_test_user") == F.lit(False))
        .select(
            F.col("u.user_id").alias("user_id"),
            F.col("u.source_system").alias("source_system"),
            F.col("g.game_category_id").alias("game_category_id"),
        )
        .dropDuplicates(["user_id", "source_system", "game_category_id"])
    )

    presence_flags_by_user = (
        fsw_presence
        .groupBy("user_id", "source_system")
        .agg(
            F.max(F.when(F.col("game_category_id") == F.lit(LIVE_CAT), F.lit(1)).otherwise(F.lit(0))).alias("plays_live"),
            F.max(F.when(F.col("game_category_id") == F.lit(SPORTS_CAT), F.lit(1)).otherwise(F.lit(0))).alias("plays_sports"),
        )
        .withColumn("plays_both", (F.col("plays_live") * F.col("plays_sports")).cast("int"))
    )

    cluster_play_pcts = (
        predicted_df.alias("p")
        .join(
            presence_flags_by_user.alias("f"),
            (F.col("p.unique_user_id") == F.concat_ws("-", F.col("f.source_system"), F.col("f.user_id"))),
            "left"
        )
        .groupBy("cluster_id")
        .agg(
            F.avg(F.coalesce(F.col("plays_live"), F.lit(0))).alias("pct_live"),
            F.avg(F.coalesce(F.col("plays_sports"), F.lit(0))).alias("pct_sports"),
            F.avg(F.coalesce(F.col("plays_both"), F.lit(0))).alias("pct_both"),
        )
    )


    # ======================================================
    # REVENUE METRICS (unchanged, still fixed 3m window via date_sk)
    # ======================================================
    fsw_raw = settled_wager
    fsw_3m = _apply_start_filter(fsw_raw, [], [])

    wagers_by_user_cat = (
        fsw_3m
        .join(
            dim_game,
            (fsw_3m.game_sk == dim_game.game_sk) & (fsw_3m.source_system == dim_game.source_system),
            "inner"
        )
        .drop(dim_game.source_system, dim_game.game_sk)
        .filter(NON_LOTTERY_FILTER)
        .filter(CAT_ALLOCATION_FILTER)
        .groupBy("user_sk", "source_system", "game_category_id")
        .agg(F.sum(F.coalesce(F.col("wager_amount").cast("double"), F.lit(0.0))).alias("wager_amount"))
    )



    wagers_total_by_user = (
        wagers_by_user_cat
        .groupBy("user_sk", "source_system")
        .agg(F.sum(F.col("wager_amount")).alias("wager_amount_total_month"))
    )

    dep_raw = deposit
    dep_3m = _apply_start_filter(dep_raw, [], [])

    deposits_by_user = (
        dep_3m
        .filter(NON_LOTTERY_FILTER)
        .groupBy("user_sk", "source_system")
        .agg(
            F.sum(F.coalesce(F.col("deposit_amount_usd").cast("double"), F.lit(0.0))).alias("deposit_amount_month"),
            F.sum(F.coalesce(F.col("vat_on_deposit_amount").cast("double"), F.lit(0.0))).alias("vat_amount_month"),
        )
    )

    rtp_raw = rtp
    rtp_3m = _apply_start_filter(rtp_raw, [], [])

    rtp_candidates = ["rtp_amount", "rtp", "rtp_value"]
    rtp_col = next((c for c in rtp_candidates if c in rtp_3m.columns), None)
    if rtp_col is None:
        raise ValueError(
            f"Could not find RTP column in fact_rtp. Tried {rtp_candidates}. "
            f"Available columns: {rtp_3m.columns}. Please set rtp_col manually."
        )

    rtp_by_user_cat = (
        rtp_3m
        .join(
            dim_game,
            (rtp_3m.game_sk == dim_game.game_sk) & (rtp_3m.source_system == dim_game.source_system),
            "inner"
        )
        .drop(dim_game.source_system, dim_game.game_sk)
        .filter(NON_LOTTERY_FILTER)
        .filter(CAT_ALLOCATION_FILTER)
        .groupBy("user_sk", "source_system", "game_category_id")
        .agg(F.sum(F.coalesce(F.col(rtp_col).cast("double"), F.lit(0.0))).alias("rtp_amount"))
    )

    user_month_cat_amounts = (
        wagers_by_user_cat.alias("w")
        .join(
            user.select("user_sk", "source_system", "user_id", "is_test_user").alias("du"),
            (F.col("w.user_sk") == F.col("du.user_sk")) & (F.col("w.source_system") == F.col("du.source_system")),
            "inner"
        ).drop(F.col("du.user_sk"), F.col("du.source_system"))
        .filter(F.col("du.is_test_user") == F.lit(False))
        .join(
            rtp_by_user_cat.alias("r"),
            (F.col("r.user_sk") == F.col("w.user_sk")) &
            (F.col("r.source_system") == F.col("w.source_system")) &
            (F.col("r.game_category_id") == F.col("w.game_category_id")),
            "left"
        ).drop(F.col("r.user_sk"), F.col("r.source_system"), F.col("r.game_category_id"))
        .join(
            deposits_by_user.alias("d"),
            (F.col("d.user_sk") == F.col("w.user_sk")) &
            (F.col("d.source_system") == F.col("w.source_system")),
            "left"
        ).drop(F.col("d.user_sk"), F.col("d.source_system"))
        .join(
            wagers_total_by_user.alias("t"),
            (F.col("t.user_sk") == F.col("w.user_sk")) &
            (F.col("t.source_system") == F.col("w.source_system")),
            "left"
        ).drop(F.col("t.user_sk"), F.col("t.source_system"))
        .select(
            F.col("du.user_id").alias("user_id"),
            F.col("w.user_sk").alias("user_sk"),
            F.col("w.source_system").alias("source_system"),
            F.col("w.game_category_id").alias("game_category_id"),
            F.coalesce(F.col("w.wager_amount"), F.lit(0.0)).alias("wager_amount"),
            F.coalesce(F.col("r.rtp_amount"), F.lit(0.0)).alias("rtp_amount"),
            F.coalesce(F.col("d.deposit_amount_month"), F.lit(0.0)).alias("deposit_amount_month"),
            F.coalesce(F.col("d.vat_amount_month"), F.lit(0.0)).alias("vat_amount_month"),
            F.coalesce(F.col("t.wager_amount_total_month"), F.lit(0.0)).alias("wager_amount_total_month"),
        )
    )

    alloc = (
        user_month_cat_amounts
        .withColumn(
            "share_of_month",
            F.when(F.col("wager_amount_total_month") > F.lit(0.0),
                F.col("wager_amount") / F.col("wager_amount_total_month")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "vat_allocated",
            F.when(F.col("wager_amount_total_month") > F.lit(0.0),
                F.col("vat_amount_month") * (F.col("wager_amount") / F.col("wager_amount_total_month"))).otherwise(F.lit(0.0))
        )
        .withColumn(
            "deposit_allocated",
            F.when(F.col("wager_amount_total_month") > F.lit(0.0),
                F.col("deposit_amount_month") * (F.col("wager_amount") / F.col("wager_amount_total_month"))).otherwise(F.lit(0.0))
        )
    )

    alloc_cat1_by_user = (
        alloc
        .filter(CAT_ANALYSIS_FILTER)
        .groupBy("user_id", "source_system")
        .agg(
            F.sum(F.col("deposit_allocated")).alias("deposit_allocated_to_cat1"),
            F.sum(F.col("vat_allocated")).alias("vat_allocated_to_cat1"),
            F.sum((F.col("wager_amount") - F.col("rtp_amount") - F.col("vat_allocated"))).alias("revenue_cat1"),
            F.sum(F.col("wager_amount")).alias("wager_cat1"),
        )
    )

    cluster_users = predicted_df.select("unique_user_id", "cluster_id")

    cluster_agg = (
        cluster_users.alias("c")
        .join(alloc_cat1_by_user.alias("a"), F.col("c.unique_user_id") == F.concat_ws("-", F.col("a.source_system"), F.col("a.user_id")), "left")
        .groupBy("cluster_id")
        .agg(
            F.sum(F.coalesce(F.col("a.deposit_allocated_to_cat1"), F.lit(0.0))).alias("deposit_allocated_to_cat1"),
            F.sum(F.coalesce(F.col("a.vat_allocated_to_cat1"), F.lit(0.0))).alias("vat_allocated_to_cat1"),
            F.sum(F.coalesce(F.col("a.revenue_cat1"), F.lit(0.0))).alias("total_revenue_cat1"),
            F.sum(F.coalesce(F.col("a.wager_cat1"), F.lit(0.0))).alias("total_wager_cat1"),
        )
        .fillna({"deposit_allocated_to_cat1": 0.0,
                "vat_allocated_to_cat1": 0.0,
                "total_revenue_cat1": 0.0,
                "total_wager_cat1": 0.0})
    )

    cluster_sizes = (
        predicted_df
        .groupBy("cluster_id")
        .agg(F.countDistinct("unique_user_id").alias("num_users"))
    )

    # cluster_medians = (
    #     cluster_users.alias("c")
    #     .join(alloc_cat1_by_user.alias("a"), "user_id", "inner")
    #     .select(
    #         F.col("c.cluster_id").alias("cluster_id"),
    #         F.col("a.wager_cat1").cast("double").alias("user_wager_total"),
    #         F.col("a.revenue_cat1").cast("double").alias("user_revenue_total"),
    #     )
    #     .groupBy("cluster_id")
    #     .agg(
    #         F.expr("percentile_approx(user_wager_total, 0.5, 10000)").alias("median_wager_per_user"),
    #         F.expr("percentile_approx(user_revenue_total, 0.5, 10000)").alias("median_revenue_per_user"),
    #     )
    # )

    cluster_metrics_df = (
        cluster_agg
        .join(cluster_sizes, "cluster_id", "left")
        .join(clusters_meta_df, "cluster_id", "left")
        .join(cluster_play_pcts, "cluster_id", "left")
        .fillna({"num_users": 0, "pct_live": 0.0, "pct_sports": 0.0, "pct_both": 0.0})
        .withColumn(
            "avg_wager_per_user",
            F.when(F.col("num_users") > 0, F.col("total_wager_cat1") / F.col("num_users")).otherwise(F.lit(0.0))
        )
        .withColumn(
            "avg_revenue_per_user",
            F.when(F.col("num_users") > 0, F.col("total_revenue_cat1") / F.col("num_users")).otherwise(F.lit(0.0))
        )
    )

    cluster_metrics_pdf = cluster_metrics_df.toPandas().set_index("cluster_id")
    cluster_ids = (
        cluster_metrics_pdf["num_users"]
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # ======================================================
    # ACTIVITY EVENTS (NOW SPLIT: 365d vs 90d)
    # Left: last 365 days
    # Right: last 90 days
    # ======================================================
    def _build_events(days):
        # Wagers (use timestamp col)
        wager_base = (
            _filter_last_days(wager, days=days, ts_col=wager_time_col).alias("w")
            .filter(NON_LOTTERY_FILTER)
            .filter(CAT_ANALYSIS_FILTER)
            .join(
                dim_game.select("game_sk", "source_system", "game_name").alias("dg"),
                (F.col("w.game_sk") == F.col("dg.game_sk")) & (F.col("w.source_system") == F.col("dg.source_system")),
                "left"
            ).drop(F.col("dg.game_sk"), F.col("dg.source_system"))
            .select(
                F.col("user_id").cast("string").alias("user_id"),
                F.col("wager_sk").alias("wager_sk"),
                F.col("source_system").alias("source_system"),
                F.col(f"w.{wager_time_col}").cast("timestamp").alias("event_ts"),
                F.col(f"w.{wager_stake_col}").cast("double").alias("wager_amount"),
                F.coalesce(F.col("dg.game_name"), F.lit("Unknown")).alias("game_name"),
            )
            .alias("w")
        )

        wager_n = (
            wager_base
            .join(
                predicted_df.alias("p"),
                F.col("p.unique_user_id") == F.concat_ws("-", F.col("w.source_system"), F.col("w.user_id")),
                "inner"
            )
            .select(
                F.col("p.cluster_id").alias("cluster_id"),
                F.col("w.user_id"),
                F.col("w.wager_sk"),
                F.col("w.source_system"),
                F.col("w.event_ts"),
                F.col("w.wager_amount"),
                F.col("w.game_name"),
            )
        )

        # Settled keys (ensure settled; use settlement timestamp if date_sk not present)
        # If your table has date_sk, it will use it; otherwise it will use wager_settlement_timestamp
        fsw_n = (
            _filter_last_days(settled_wager, days=days, ts_col="wager_settlement_timestamp").alias('f')
            .join(
                dim_game,
                (F.col("f.game_sk") == dim_game.game_sk) & (F.col("f.source_system") == dim_game.source_system),
                "inner"
            )
            .drop(dim_game.source_system, dim_game.game_sk)
            .filter(NON_LOTTERY_FILTER)
            .filter(CAT_ANALYSIS_FILTER)
            .select("wager_sk", "source_system")
            .dropDuplicates(["wager_sk", "source_system"])
            .alias("s")
        )

        return (
            wager_n.alias("w")
            .join(
                fsw_n,
                (F.col("w.wager_sk") == F.col("s.wager_sk")) & (F.col("w.source_system") == F.col("s.source_system")),
                "inner"
            ).drop(F.col("s.wager_sk"), F.col("s.source_system"))
        )

    events_365 = _build_events(DAYS_ALL)
    events_90  = _build_events(DAYS_RECENT)

    def _hourly_weekday(events_df):
        hourly = (
            events_df
            .withColumn("hour", F.hour("event_ts"))
            .groupBy("cluster_id", "hour")
            .agg(F.count("*").alias("cnt"))
        )
        weekday = (
            events_df
            .withColumn("dow", F.dayofweek("event_ts"))  # 1=Sun..7=Sat
            .groupBy("cluster_id", "dow")
            .agg(F.count("*").alias("cnt"))
        )
        return hourly.toPandas(), weekday.toPandas()

    hourly_365_pdf, weekday_365_pdf = _hourly_weekday(events_365)
    hourly_90_pdf,  weekday_90_pdf  = _hourly_weekday(events_90)

    TOP_GAMES_N = 10

    def _top_games_pdf(events_df, top_n):
        base = (
            events_df
            .withColumn(
                "game_name",
                F.when(
                    F.col("game_name").isNull() | (F.length(F.trim(F.col("game_name"))) == 0),
                    F.lit("Unknown")
                ).otherwise(F.col("game_name"))
            )
            .withColumn("wager_amount_clean", F.coalesce(F.col("wager_amount"), F.lit(0.0)))
            .groupBy("cluster_id", "game_name")
            .agg(F.sum("wager_amount_clean").alias("wager_amount_sum"))
        )

        w = Window.partitionBy("cluster_id").orderBy(
            F.col("wager_amount_sum").desc(),
            F.col("game_name").asc()
        )

        top = (
            base
            .withColumn("rn", F.row_number().over(w))
            .filter(F.col("rn") <= F.lit(int(top_n)))
            .drop("rn")
        )
        return top.toPandas()

    top_games_365_pdf = _top_games_pdf(events_365, TOP_GAMES_N)
    top_games_90_pdf  = _top_games_pdf(events_90,  TOP_GAMES_N)

    # ======================================================
    # AGE + NATIONALITY (same as before)
    # ======================================================
    uv = spark.table("prod_silver.user.user_verification").alias("uv")

    age_df = (
        predicted_df.alias("pred")
        .join(
            uv,
            (F.col("pred.unique_user_id") == F.concat_ws("-", F.col("uv.source_system"), F.col("uv.user_id"))) &
            (F.col("uv.verified_status") == F.lit("3")) &
            (F.col("uv.verify_type") == F.lit("3")),
            "left"
        ).drop(F.col("uv.user_id"))
        .withColumn(
            "birth_date_clean",
            F.when(
                (F.length(F.trim(F.col("uv.birth_date"))) == 0) | (F.length(F.trim(F.col("uv.birth_date"))) > 10),
                F.lit(None)
            ).otherwise(F.col("uv.birth_date"))
        )
        .withColumn("birth_date_dt", F.to_date("birth_date_clean"))
        .withColumn(
            "age",
            F.when(
                F.col("birth_date_dt").isNull(),
                F.lit(None)
            ).otherwise(
                F.floor(F.months_between(F.current_date(), F.col("birth_date_dt")) / F.lit(12.0)).cast("int")
            )
        )
        .select(
            F.col("pred.cluster_id").alias("cluster_id"),
            F.col("pred.unique_user_id").alias("user_id"),
            F.col("age").alias("age")
        )
    )

    age_hist_pdf = (
        age_df
        .filter(F.col("age").isNotNull())
        .filter((F.col("age") >= 0) & (F.col("age") <= 110))
        .groupBy("cluster_id", "age")
        .agg(F.count(F.lit(1)).alias("cnt"))
        .toPandas()
    )

    NATIONALITY_COL_CANDIDATES = ["country_region", "nationality", "country", "country_name"]
    nat_col = next((c for c in NATIONALITY_COL_CANDIDATES if c in uv.columns), None)
    if nat_col is None:
        raise ValueError(
            f"Could not find nationality column in prod_silver.user.user_verification. "
            f"Tried {NATIONALITY_COL_CANDIDATES}. Available: {uv.columns}"
        )

    nationality_df = (
        predicted_df.alias("pred")
        .join(
            uv,
            (F.col("pred.unique_user_id") == F.concat_ws("-", F.col("uv.source_system"), F.col("uv.user_id"))) &
            (F.col("uv.verified_status") == F.lit("3")) &
            (F.col("uv.verify_type") == F.lit("3")),
            "left"
        ).drop(F.col("pred.unique_user_id"))
        .withColumn(
            "nationality",
            F.when(
                F.col(f"uv.{nat_col}").isNull() | (F.length(F.trim(F.col(f"uv.{nat_col}"))) == 0),
                F.lit("Unknown")
            ).otherwise(F.col(f"uv.{nat_col}"))
        )
        .groupBy("cluster_id", "nationality")
        .agg(F.count(F.lit(1)).alias("cnt"))
    )

    nat_pdf = (
        nationality_df
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("cluster_id").orderBy(F.col("cnt").desc(), F.col("nationality").asc())
            )
        )
        .withColumn("nationality_bucket", F.when(F.col("rn") <= 5, F.col("nationality")).otherwise(F.lit("Other")))
        .groupBy("cluster_id", "nationality_bucket")
        .agg(F.sum("cnt").alias("cnt"))
        .toPandas()
    )

    # ======================================================
    # Plotly helpers
    # ======================================================
    def _plotly_bar(x, y, title, height=220):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, marker=dict(opacity=0.85)))
        fig.update_layout(
            template="plotly_dark",
            margin=dict(t=40, l=10, r=10, b=10),
            height=height,
            title=dict(text=title, x=0.02, y=0.95, font=dict(size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E6EDF3"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.10)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.10)", zeroline=False),
        )
        fig.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
        return fig

    DOW_LABELS = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _hour_series_for_cluster(cid, hourly_pdf):
        df = hourly_pdf[hourly_pdf["cluster_id"] == cid]
        m = {int(r["hour"]): int(r["cnt"]) for _, r in df.iterrows()} if not df.empty else {}
        hours = list(range(0, 24))
        counts = [m.get(h, 0) for h in hours]
        return hours, counts

    def _dow_series_for_cluster(cid, weekday_pdf):
        df = weekday_pdf[weekday_pdf["cluster_id"] == cid]
        m = {DOW_LABELS.get(int(r["dow"]), str(int(r["dow"]))): int(r["cnt"]) for _, r in df.iterrows()} if not df.empty else {}
        labels = DOW_ORDER
        counts = [m.get(lbl, 0) for lbl in labels]
        return labels, counts

    def _age_series_for_cluster(cid):
        df = age_hist_pdf[age_hist_pdf["cluster_id"] == cid]
        if df.empty:
            return [], []
        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["age"])
        if df.empty:
            return [], []
        df["age"] = df["age"].astype(int)
        df = df.sort_values("age")
        return df["age"].tolist(), df["cnt"].tolist()

    def _nat_series_for_cluster(cid):
        df = nat_pdf[nat_pdf["cluster_id"] == cid]
        if df.empty:
            return [], []
        df = df.copy()
        df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values("cnt", ascending=False)
        return df["nationality_bucket"].astype(str).tolist(), df["cnt"].tolist()

    def _top_games_series_for_cluster(cid, top_games_pdf):
        df = top_games_pdf[top_games_pdf["cluster_id"] == cid]
        if df.empty:
            return [], []
        df = df.copy()
        df["wager_amount_sum"] = pd.to_numeric(df["wager_amount_sum"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["wager_amount_sum", "game_name"], ascending=[False, True])
        return df["game_name"].astype(str).tolist(), df["wager_amount_sum"].tolist()

    def _fmt_money(x):
        try:
            return f"{float(x):,.2f}"
        except:
            return "0.00"

    def _fmt_int(x):
        try:
            return f"{int(x):,}"
        except:
            return "0"
        
    def _fmt_pct(x):
        try:
            return f"{float(x) * 100:.1f}%"
        except:
            return "0.0%"

    # ======================================================
    # Global bar chart (X=cluster name, Y=total_revenue)
    # ======================================================
    bar_pdf = cluster_metrics_df.select("cluster_id", "total_revenue_cat1", "cluster_name").toPandas()
    bar_pdf["cluster_name"] = bar_pdf["cluster_name"].fillna("").replace("", np.nan)
    bar_pdf["cluster_name"] = bar_pdf["cluster_name"].fillna(bar_pdf["cluster_id"])
    bar_pdf["total_revenue_cat1"] = pd.to_numeric(bar_pdf["total_revenue_cat1"], errors="coerce").fillna(0.0)
    bar_pdf = bar_pdf.sort_values("total_revenue_cat1", ascending=False)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=bar_pdf["cluster_name"].tolist(), y=bar_pdf["total_revenue_cat1"].tolist(), marker=dict(opacity=0.90)))
    fig_rev.update_layout(
        template="plotly_dark",
        margin=dict(t=50, l=10, r=10, b=60),
        height=340,
        title=dict(text="Revenue — last 3 months — by cluster", x=0.02, y=0.95, font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        xaxis=dict(title="Cluster", showgrid=False, tickangle=-25),
        yaxis=dict(title="Revenue", showgrid=True, gridcolor="rgba(255,255,255,0.10)", zeroline=False),
    )
    fig_rev.update_traces(hovertemplate="%{x}<br>Revenue: %{y}<extra></extra>")
    rev_bar_html = pio.to_html(fig_rev, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

    # ======================================================
    # Build cluster cards HTML
    # ======================================================
    PLOTLY_JS = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    TITLE_LEFT  = f"Activity patterns — last {DAYS_ALL} days"
    TITLE_RIGHT = f"Activity patterns — last {DAYS_RECENT} days"

    cluster_cards = []
    for cid in cluster_ids:
        row = cluster_metrics_pdf.loc[cid]
        name = row.get("cluster_name", cid) if pd.notna(row.get("cluster_name", None)) else cid
        desc = row.get("cluster_description", "") if pd.notna(row.get("cluster_description", None)) else ""
        n_users = _fmt_int(row.get("num_users", 0))

        # median_wager = _fmt_money(row.get("median_wager_per_user", np.nan))
        # median_revenue = _fmt_money(row.get("median_revenue_per_user", np.nan))
        avg_wager = _fmt_money(row.get("avg_wager_per_user", np.nan))
        avg_revenue = _fmt_money(row.get("avg_revenue_per_user", np.nan))
        total_rev = _fmt_money(row.get("total_revenue_cat1", 0.0))

        pct_live   = _fmt_pct(row.get("pct_live", 0.0))
        pct_sports = _fmt_pct(row.get("pct_sports", 0.0))
        pct_both   = _fmt_pct(row.get("pct_both", 0.0))

        # LEFT: 365d
        hx_365, hy_365 = _hour_series_for_cluster(cid, hourly_365_pdf)
        dx_365, dy_365 = _dow_series_for_cluster(cid, weekday_365_pdf)
        gx_365, gy_365 = _top_games_series_for_cluster(cid, top_games_365_pdf)

        # RIGHT: 90d
        hx_90, hy_90 = _hour_series_for_cluster(cid, hourly_90_pdf)
        dx_90, dy_90 = _dow_series_for_cluster(cid, weekday_90_pdf)
        gx_90,  gy_90  = _top_games_series_for_cluster(cid, top_games_90_pdf)

        fig_hour_365 = _plotly_bar(hx_365, hy_365, "Daily activity (by hour)")
        fig_dow_365  = _plotly_bar(dx_365, dy_365, "Weekly activity (by day)")
        fig_hour_90  = _plotly_bar(hx_90,  hy_90,  "Daily activity (by hour)")
        fig_dow_90   = _plotly_bar(dx_90,  dy_90,  "Weekly activity (by day)")

        if len(gx_365) > 0:
            fig_games_365 = _plotly_bar(gx_365, gy_365, "Top games (wager amount)", height=260)
            fig_games_365.update_layout(xaxis=dict(tickangle=-25))
            games_365_html = pio.to_html(fig_games_365, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
            games_365_note = ""
        else:
            games_365_html = ""
            games_365_note = '<div class="muted" style="margin-top:8px;">No game data for this cluster (365d).</div>'

        if len(gx_90) > 0:
            fig_games_90 = _plotly_bar(gx_90, gy_90, "Top games (wager amount)", height=260)
            fig_games_90.update_layout(xaxis=dict(tickangle=-25))
            games_90_html = pio.to_html(fig_games_90, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
            games_90_note = ""
        else:
            games_90_html = ""
            games_90_note = '<div class="muted" style="margin-top:8px;">No game data for this cluster (90d).</div>'

        hour_365_html = pio.to_html(fig_hour_365, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
        dow_365_html  = pio.to_html(fig_dow_365,  include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
        hour_90_html  = pio.to_html(fig_hour_90,  include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
        dow_90_html   = pio.to_html(fig_dow_90,   include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

        # Age + Nationality
        ax, ay = _age_series_for_cluster(cid)
        nx, ny = _nat_series_for_cluster(cid)

        if len(ax) > 0:
            fig_age = _plotly_bar(ax, ay, "Age distribution (KYC verified)", height=220)
            age_html = pio.to_html(fig_age, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
            age_note = ""
        else:
            age_html = ""
            age_note = '<div class="muted" style="margin-top:8px;">No valid birth_date found for this cluster.</div>'

        if len(nx) > 0:
            fig_nat = _plotly_bar(nx, ny, "Nationality (Top 5 + Other)", height=220)
            nat_html = pio.to_html(fig_nat, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
            nat_note = ""
        else:
            nat_html = ""
            nat_note = '<div class="muted" style="margin-top:8px;">No nationality info found for this cluster.</div>'

        cluster_cards.append(f"""
        <details class="card">
        <summary>
            <div class="card-title">
            <span class="dot"></span>
            <span class="cname">{name}</span>
            </div>
            <div class="chip">n={n_users}</div>
        </summary>

        <div class="card-body">
            <div class="muted">{desc}</div>

            <div class="kpis">
            <div class="kpi">
                <div class="kpi-label">Avg wager amount / user (last 90 days)</div>
                <div class="kpi-value">${avg_wager}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">ARPU (last 90 days)</div>
                <div class="kpi-value">${avg_revenue}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Total revenue (last 90 days)</div>
                <div class="kpi-value">${total_rev}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">% users also play Live Casino</div>
                <div class="kpi-value">{pct_live}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">% users also play Sports</div>
                <div class="kpi-value">{pct_sports}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">% users play both (Live + Sports)</div>
                <div class="kpi-value">{pct_both}</div>
            </div>
            </div>

            <div class="activity-two">
            <div class="activity-col">
                <div class="section-title">{TITLE_LEFT}</div>
                <div class="plots">
                <div class="plot">{hour_365_html}</div>
                <div class="plot">{dow_365_html}</div>
                </div>
            </div>

            <div class="activity-col">
                <div class="section-title">{TITLE_RIGHT}</div>
                <div class="plots">
                <div class="plot">{hour_90_html}</div>
                <div class="plot">{dow_90_html}</div>
                </div>
            </div>
            </div>

            <div class="activity-two">
                <div class="activity-col">
                    <div class="section-title">Top games — last 365 days</div>
                    <div class="plots onecol">
                    <div class="plot">{games_365_html}{games_365_note}</div>
                    </div>
                </div>

                <div class="activity-col">
                    <div class="section-title">Top games — last 90 days</div>
                    <div class="plots onecol">
                    <div class="plot">{games_90_html}{games_90_note}</div>
                    </div>
                </div>
            </div>

            <div class="plots">
            <div class="plot">{age_html}{age_note}</div>
            <div class="plot">{nat_html}{nat_note}</div>
            </div>
        </div>
        </details>
        """)

    clusters_html = "\n".join(cluster_cards)

    # ======================================================
    # FINAL HTML
    # ======================================================
    html = f"""
    {PLOTLY_JS}

    <div class="report" data-theme="dark">
    <div class="header">
        <div>
        <div class="title">Cluster Segmentation — {analysis_name}</div>
        </div>
        <div class="badges">
        <span class="badge">{len(cluster_ids)} clusters</span>
        <span class="badge">Revenue window: fixed 3 months</span>
        <span class="badge">Activity windows: {DAYS_ALL}d vs {DAYS_RECENT}d</span>
        </div>
    </div>

    <div class="panel span2">
        <div class="panel-title">Revenue overview</div>
        <div class="plot bigplot">{rev_bar_html}</div>
    </div>

    <div class="panel span2">
        <div class="panel-title">Clusters</div>
        <div class="panel-subtitle">
        Ordered by <b>num_users (descending)</b>. Open each cluster to view KPIs and activity distributions.
        </div>
        <div class="cards">{clusters_html}</div>
    </div>
    </div>

    <style>
    :root {{
        --bg: #0E1117;
        --panel: #161B22;
        --text: #E6EDF3;
        --muted: #9BA3AF;
        --border: rgba(255,255,255,0.08);
        --shadow: rgba(0,0,0,0.45);
        --chip: rgba(255,255,255,0.08);
    }}

    .report, .report * {{ box-sizing: border-box; }}
    .report {{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        background: var(--bg);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
    }}

    .header {{
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }}
    .title {{ font-size: 22px; font-weight: 800; letter-spacing: 0.2px; line-height: 1.2; }}

    .badges {{ display:flex; gap: 8px; flex-wrap: wrap; align-items:center; }}
    .badge {{
        font-size: 12px;
        color: var(--muted);
        border: 1px solid var(--border);
        background: var(--chip);
        padding: 6px 10px;
        border-radius: 999px;
        white-space: nowrap;
    }}

    .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 10px 25px var(--shadow);
        padding: 14px;
        min-width: 0;
        margin-bottom: 14px;
    }}

    .panel-title {{ font-weight: 800; font-size: 14px; }}
    .panel-subtitle {{ font-size: 12px; color: var(--muted); margin-top: 4px; margin-bottom: 10px; line-height: 1.35; }}

    .cards {{ display: grid; gap: 10px; }}

    details.card {{
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--panel);
        overflow: hidden;
    }}
    details.card > summary {{
        cursor: pointer;
        list-style: none;
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding: 12px 12px;
        gap: 12px;
    }}
    details.card[open] > summary {{
        background: var(--chip);
        border-bottom: 1px solid var(--border);
    }}
    summary::-webkit-details-marker {{ display:none; }}

    .card-title {{ display:flex; align-items:baseline; gap: 8px; min-width: 0; }}
    .dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; flex: 0 0 auto; background: #2E6F9E; }}
    .cname {{
        color: var(--text);
        font-weight: 800;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 70vw;
    }}
    .chip {{
        font-size: 12px;
        color: var(--muted);
        border: 1px solid var(--border);
        background: var(--chip);
        padding: 5px 9px;
        border-radius: 999px;
        white-space: nowrap;
        flex: 0 0 auto;
    }}
    .card-body {{ padding: 12px; }}
    .muted {{ font-size: 12px; color: var(--muted); line-height: 1.35; }}

    .kpis {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .kpi {{
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 10px;
        min-width: 0;
    }}
    .kpi-label {{
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 0.2px;
    }}

    .section-title {{
        font-size: 12px;
        font-weight: 900;
        color: var(--text);
        margin: 10px 0 8px 0;
        letter-spacing: 0.2px;
    }}

    .plots {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 10px;
    }}
    .plots.onecol {{
        grid-template-columns: 1fr;
    }}

    .plot {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
        padding: 6px;
    }}

    .activity-two {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 8px;
    }}
    .activity-col {{
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
        padding: 10px;
    }}

    .bigplot {{ padding: 10px; }}

    @media (max-width: 1100px) {{
        .kpis {{ grid-template-columns: 1fr; }}
        .plots {{ grid-template-columns: 1fr; }}
        .activity-two {{ grid-template-columns: 1fr; }}
        .cname {{ max-width: 85vw; }}
    }}

    html, body {{ background: #000 !important; }}
    #app,
    #app > div,
    .notebook-container,
    .notebook-content,
    div[data-testid="notebook-container"],
    .output,
    .output_area,
    .cell-output,
    .cell-output-area {{ background: #000 !important; }}
    .displayedHTML,
    .displayedHTML > div {{ background: #000 !important; }}
    </style>
    """

    return html

# display(fact_settled_wager)


# COMMAND ----------


def get_clusters_report_detailed_information(analysis: str):
    html = get_html_cluster_detailed_information(analysis)
    displayHTML(html)
    save_report(analysis=analysis, html=html, report_name="detailed_information")

# COMMAND ----------

for analysis, analysis_name in ANALYSIS_NAMES.items():
    get_clusters_report_detailed_information(analysis=analysis)