# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 – Evaluation & Report (Lottery Net Turnover Prediction)
# MAGIC
# MAGIC Backtests the champion model on the last N weeks of held-out history.
# MAGIC
# MAGIC **Charts:**
# MAGIC 1. Actual vs Predicted Weekly Revenue (bar per fold)
# MAGIC 2. MAPE per fold (bar with threshold line)
# MAGIC 3. Residuals over time (bar)

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_lottery_net_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "model_alias", "model_catalog", "mape_threshold", "n_eval_folds"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id",    "analysis_1")
dbutils.widgets.text("model_alias",    "champion")
dbutils.widgets.text("model_catalog",  "ml_train")
dbutils.widgets.text("mape_threshold", "15.0")
dbutils.widgets.text("n_eval_folds",   "12")

# COMMAND ----------

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

analysis_id    = dbutils.widgets.get("analysis_id").strip()
model_alias    = dbutils.widgets.get("model_alias").strip()
model_catalog  = dbutils.widgets.get("model_catalog").strip()
mape_threshold = float(dbutils.widgets.get("mape_threshold").strip())
n_eval_folds   = int(dbutils.widgets.get("n_eval_folds").strip())
ANALYSIS_NAME  = ANALYSIS_NAMES[analysis_id]

FEATURE_COLS = [
    "revenue_lag_1w",
    "revenue_lag_4w",
    "revenue_lag_8w",
    "revenue_lag_12w",
    "revenue_rolling_4w_avg",
    "revenue_rolling_4w_std",
    "revenue_rolling_8w_avg",
    "revenue_rolling_12w_avg",
    "revenue_trend_4w",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_month_start",
    "is_month_end",
    "is_summer",
    "is_winter",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "is_ramadan",
    "is_eid",
    "is_uae_national_day",
    "has_payday",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load model and training features

# COMMAND ----------

model = load_model(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=model_catalog,
    alias=model_alias,
)

train_spark = read_current_table(
    spark=spark,
    table_base=f"{T_LOTTERY_FEATURES_PREFIX}_{ANALYSIS_NAME}_train",
)

for col in [f.name for f in train_spark.schema.fields if isinstance(f.dataType, DecimalType)]:
    train_spark = train_spark.withColumn(col, F.col(col).cast("double"))

train_pd = train_spark.orderBy("week_start").toPandas().reset_index(drop=True)
print(f"Loaded {len(train_pd):,} weekly rows")
print(f"Date range: {train_pd['week_start'].min()} → {train_pd['week_start'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Backtest: last N folds (one prediction per holdout week)

# COMMAND ----------

avail_cols   = [c for c in FEATURE_COLS if c in train_pd.columns]
n_total      = len(train_pd)
n_eval_folds = min(n_eval_folds, n_total - 13)

# Convert to numpy upfront to avoid pandas groupby SIGSEGV inside sklearn
X_all     = train_pd[avail_cols].astype("float64").values
y_all     = train_pd["revenue"].astype("float64").values
weeks_all = train_pd["week_start"].astype(str).values

fold_results = []

for i in range(n_eval_folds, 0, -1):
    test_idx = n_total - i
    X_te     = X_all[[test_idx]]
    y_te     = float(y_all[test_idx])
    y_pred   = float(np.clip(model.predict(X_te), 0, None)[0])
    mape     = abs(y_te - y_pred) / y_te * 100 if y_te > 0 else np.nan

    fold_results.append({
        "target_week":       str(weeks_all[test_idx])[:10],
        "actual_revenue":    y_te,
        "predicted_revenue": y_pred,
        "residual":          y_pred - y_te,
        "mape":              mape,
    })

results_df   = pd.DataFrame(fold_results)
overall_mape = float(results_df["mape"].mean())
weeks        = results_df["target_week"].tolist()

print(f"\nOverall MAPE: {overall_mape:.2f}%  (threshold: {mape_threshold:.1f}%,  folds: {len(results_df)})")
print(results_df[["target_week", "actual_revenue", "predicted_revenue", "mape"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 1 – Actual vs Predicted Weekly Revenue

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 5))
x, w = np.arange(len(results_df)), 0.35

ax.bar(x - w/2, results_df["actual_revenue"],    w, label="Actual",    color="#2196F3", alpha=0.85)
ax.bar(x + w/2, results_df["predicted_revenue"], w, label="Predicted", color="#FF9800", alpha=0.85)
ax.set_title(f"Actual vs Predicted Weekly Lottery Revenue — {ANALYSIS_NAME}", fontsize=13, fontweight="bold")
ax.set_xlabel("Week")
ax.set_ylabel("Net Turnover (native)")
ax.set_xticks(x)
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 2 – MAPE per Fold

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4))
colors = ["#4CAF50" if m < mape_threshold else "#F44336" for m in results_df["mape"]]

ax.bar(weeks, results_df["mape"], color=colors, alpha=0.85)
ax.axhline(mape_threshold, color="#F44336", linestyle="--", linewidth=1.5,
           label=f"Threshold {mape_threshold:.0f}%")
ax.axhline(overall_mape,   color="#333333", linestyle="-",  linewidth=1.5,
           label=f"Mean MAPE {overall_mape:.1f}%")
ax.set_title(f"Lottery Revenue MAPE per Fold — {ANALYSIS_NAME}", fontsize=13, fontweight="bold")
ax.set_xlabel("Week")
ax.set_ylabel("MAPE (%)")
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(weeks, results_df["actual_revenue"],    marker="o", label="Actual Revenue",
        color="#4C72B0", linewidth=2)
ax.plot(weeks, results_df["predicted_revenue"], marker="s", label="Predicted Revenue",
        color="#DD8452", linewidth=2, linestyle="--")

ax.set_title(f"Actual vs Predicted Weekly Lottery Revenue\n({ANALYSIS_NAME}, alias={model_alias})",
             fontsize=13, pad=12)
ax.set_xlabel("Week")
ax.set_ylabel("Net Turnover (native)")
ax.set_xticks(range(len(weeks)))
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 3 – Residuals over Time

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4))
colors_res = ["#F44336" if r > 0 else "#2196F3" for r in results_df["residual"]]
ax.bar(weeks, results_df["residual"], color=colors_res, alpha=0.85)
ax.axhline(0, color="#333333", linewidth=1.0)
ax.set_title(f"Prediction Residuals (Predicted − Actual) — {ANALYSIS_NAME}", fontsize=13, fontweight="bold")
ax.set_xlabel("Week")
ax.set_ylabel("Residual (native)")
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Persist evaluation & write HTML report

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {REPORT_SPACE}")

eval_spark = (
    spark.createDataFrame(results_df)
    .withColumn("model_alias",    F.lit(model_alias))
    .withColumn("overall_mape",   F.lit(overall_mape))
    .withColumn("mape_threshold", F.lit(mape_threshold))
    .withColumn("analysis_name",  F.lit(ANALYSIS_NAME))
    .withColumn("created_at",     F.current_timestamp())
)

publish_table(
    spark=spark,
    df=eval_spark,
    table_base=f"{T_LOTTERY_SUMMARY_PREFIX}_{ANALYSIS_NAME}_eval",
    analysis_name=f"{ANALYSIS_NAME}_eval",
    current_mode="view",
    primary_keys=["target_week", "run_ts"],
    timeseries_column="run_ts",
)

dbutils.jobs.taskValues.set(key="overall_mape", value=round(overall_mape, 2))

# COMMAND ----------

# HTML report
generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
status_color = "#16a34a" if overall_mape <= mape_threshold else "#dc2626"
status_label = "PASS" if overall_mape <= mape_threshold else "FAIL"

rows_html = "\n".join(
    f'<tr><td>{r["target_week"]}</td>'
    f'<td style="text-align:right">{r["actual_revenue"]:,.0f}</td>'
    f'<td style="text-align:right">{r["predicted_revenue"]:,.0f}</td>'
    f'<td style="text-align:right;color:{"#16a34a" if r["mape"] < mape_threshold else "#dc2626"}">{r["mape"]:.1f}%</td>'
    f'<td style="text-align:right">{r["residual"]:,.0f}</td></tr>'
    for _, r in results_df.iterrows()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Lottery Net Turnover – Evaluation – {ANALYSIS_NAME}</title>
  <style>
    body{{font-family:'Trebuchet MS',sans-serif;padding:32px;background:#f4f6f8;color:#1f2a37}}
    h1{{margin-bottom:4px}} .meta{{color:#64748b;font-size:13px;margin-bottom:24px}}
    .kpi{{display:flex;gap:14px;margin-bottom:24px}}
    .card{{background:#fff;border-radius:14px;padding:16px 20px;box-shadow:0 4px 12px rgba(0,0,0,.08);min-width:160px}}
    .label{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:4px}}
    .value{{font-size:28px;font-weight:700;color:#0f766e}}
    table{{border-collapse:collapse;width:100%;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,.08)}}
    th{{background:#0f766e;color:#fff;padding:10px 14px;text-align:left;font-size:12px;text-transform:uppercase}}
    td{{padding:9px 14px;border-bottom:1px solid #f0f0f0;font-size:13px}}
    tr:last-child td{{border-bottom:none}}
  </style>
</head>
<body>
  <h1>Lottery Net Turnover – Backtest Evaluation</h1>
  <div class="meta">analysis: <strong>{ANALYSIS_NAME}</strong> | alias: {model_alias} | generated: {generated_at}</div>
  <div class="kpi">
    <div class="card"><div class="label">Overall MAPE</div><div class="value" style="color:{status_color}">{overall_mape:.1f}%</div></div>
    <div class="card"><div class="label">Status</div><div class="value" style="color:{status_color}">{status_label}</div></div>
    <div class="card"><div class="label">Threshold</div><div class="value">{mape_threshold:.0f}%</div></div>
    <div class="card"><div class="label">Folds</div><div class="value">{len(results_df)}</div></div>
  </div>
  <table>
    <tr><th>Week</th><th>Actual</th><th>Predicted</th><th>MAPE</th><th>Residual</th></tr>
    {rows_html}
  </table>
</body>
</html>"""

report_path = f"{REPORT_DIR}/lottery_net_turnover_report_{ANALYSIS_NAME}.html"
write_info  = write_report_with_single_backup(report_path, html)
print(f"Report written: {write_info['report_path']}")

print(json.dumps({
    "overall_mape":   overall_mape,
    "mape_threshold": mape_threshold,
    "status":         status_label,
    "num_folds":      len(results_df),
}, indent=2))
