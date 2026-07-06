# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 – Evaluation: Backtest & MAPE (Active User Prediction)
# MAGIC
# MAGIC Runs the champion model against every historical week fold in the training set
# MAGIC to produce backtest charts and MAPE metrics.
# MAGIC
# MAGIC **Backtest logic per fold:**
# MAGIC ```
# MAGIC actual_active_this_week   = users with label = 1 in that fold
# MAGIC predicted_active_this_week = active_last_week - predicted_churn
# MAGIC MAPE (fold) = |actual - predicted| / actual × 100
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_active_user_prediction

# COMMAND ----------

for _w in ["analysis_id", "model_alias", "churn_threshold"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id",     "analysis_4")
dbutils.widgets.text("model_alias",     "champion")
dbutils.widgets.text("churn_threshold", "0.25")

# COMMAND ----------

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timezone
from pyspark.sql.types import DecimalType
from pyspark.sql import functions as F

analysis_id     = dbutils.widgets.get("analysis_id").strip()
model_alias     = dbutils.widgets.get("model_alias").strip()
churn_threshold = float(dbutils.widgets.get("churn_threshold").strip())

ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]

FEATURE_COLS = [
    "num_bets_total",
    "num_active_days",
    "num_active_weeks",
    "days_since_last_bet",
    "avg_bets_per_active_day",
    "num_bets_90d",
    "active_days_90d",
    "active_weeks_90d",
    "stake_sum_90d",
    "stake_avg_90d",
    "bets_last_4w",
    "bets_prior_4w",
    "bet_count_trend",
    "stake_sum_total",
    "stake_cv",
    "win_rate",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load champion model and training features

# COMMAND ----------

model = load_model(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=WRITE_CATALOG,
    alias=model_alias,
)

train_table_base = f"{T_ACTIVE_USER_FEATURES_PREFIX}_{ANALYSIS_NAME}_train"
train_spark = read_current_table(spark=spark, table_base=train_table_base)

# Cast Decimal → double before toPandas
decimal_cols = [f.name for f in train_spark.schema.fields if isinstance(f.dataType, DecimalType)]
for c in decimal_cols:
    train_spark = train_spark.withColumn(c, F.col(c).cast("double"))

train_pd = train_spark.toPandas().reset_index(drop=True)
train_pd[FEATURE_COLS] = train_pd[FEATURE_COLS].astype("float64")

print(f"Loaded {len(train_pd):,} rows across {train_pd['fold'].nunique()} historical folds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run backtest: predict on every historical fold

# COMMAND ----------

# Use numpy-based fold iteration to avoid pandas groupby SIGSEGV inside sklearn
available_feat_cols = [c for c in FEATURE_COLS if c in train_pd.columns]
fold_arr      = train_pd["fold"].values
label_arr     = train_pd["label"].values
week_end_arr  = train_pd["ref_week_end"].values
X_all         = train_pd[available_feat_cols].values

records = []

for fold in np.unique(fold_arr):
    mask   = fold_arr == fold
    X_fold = X_all[mask]
    labels = label_arr[mask]

    proba       = model.predict_proba(X_fold)
    prob_churn  = proba[:, 0]
    pred_churn  = (prob_churn >= churn_threshold).astype(int)

    active_last_week      = int(mask.sum())
    actual_active         = int(labels.sum())
    predicted_churn_count = int(pred_churn.sum())
    predicted_active      = active_last_week - predicted_churn_count

    mape = abs(actual_active - predicted_active) / actual_active * 100 if actual_active > 0 else np.nan

    records.append({
        "fold":                 int(fold),
        "week_end":             week_end_arr[mask][0],
        "active_last_week":     active_last_week,
        "actual_active":        actual_active,
        "predicted_active":     predicted_active,
        "actual_churn":         active_last_week - actual_active,
        "predicted_churn":      predicted_churn_count,
        "actual_churn_rate":    (active_last_week - actual_active) / active_last_week,
        "predicted_churn_rate": predicted_churn_count / active_last_week,
        "mape":                 mape,
    })

backtest_df = pd.DataFrame(records).sort_values("week_end").reset_index(drop=True)
overall_mape = backtest_df["mape"].mean()

print(f"Overall MAPE: {overall_mape:.2f}%")
display(backtest_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 1 – Actual vs Predicted Active Users (Backtest)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 5))

weeks = backtest_df["week_end"].astype(str)
x     = np.arange(len(weeks))
width = 0.35

bars_actual    = ax.bar(x - width / 2, backtest_df["actual_active"],    width, label="Actual Active",    color="#4C72B0", alpha=0.85)
bars_predicted = ax.bar(x + width / 2, backtest_df["predicted_active"], width, label="Predicted Active", color="#DD8452", alpha=0.85)

ax.set_title(f"Backtest – Actual vs Predicted Active Users\n({ANALYSIS_NAME}, alias={model_alias})", fontsize=13, pad=12)
ax.set_xlabel("Week Ending")
ax.set_ylabel("Active Users")
ax.set_xticks(x)
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 2 – MAPE per Week

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4))

colors = ["#d62728" if m > 10 else "#2ca02c" for m in backtest_df["mape"]]
ax.bar(weeks, backtest_df["mape"], color=colors, alpha=0.85)
ax.axhline(overall_mape, color="navy", linestyle="--", linewidth=1.5, label=f"Mean MAPE = {overall_mape:.1f}%")
ax.axhline(10, color="orange", linestyle=":", linewidth=1.2, label="10% threshold")

ax.set_title(f"MAPE by Week\n({ANALYSIS_NAME}, alias={model_alias})", fontsize=13, pad=12)
ax.set_xlabel("Week Ending")
ax.set_ylabel("MAPE (%)")
ax.set_xticks(range(len(weeks)))
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chart 3 – Actual vs Predicted Churn Rate

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(weeks, backtest_df["actual_churn_rate"]    * 100, marker="o", label="Actual Churn Rate",    color="#4C72B0", linewidth=2)
ax.plot(weeks, backtest_df["predicted_churn_rate"] * 100, marker="s", label="Predicted Churn Rate", color="#DD8452", linewidth=2, linestyle="--")

ax.set_title(f"Actual vs Predicted Weekly Churn Rate\n({ANALYSIS_NAME}, alias={model_alias})", fontsize=13, pad=12)
ax.set_xlabel("Week Ending")
ax.set_ylabel("Churn Rate (%)")
ax.set_xticks(range(len(weeks)))
ax.set_xticklabels(weeks, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary table

# COMMAND ----------

summary = pd.DataFrame([{
    "analysis_name":      ANALYSIS_NAME,
    "model_alias":        model_alias,
    "churn_threshold":    churn_threshold,
    "num_folds":          len(backtest_df),
    "overall_mape":       round(overall_mape, 2),
    "best_mape":          round(backtest_df["mape"].min(), 2),
    "worst_mape":         round(backtest_df["mape"].max(), 2),
    "pct_folds_under_10": round((backtest_df["mape"] < 10).mean() * 100, 1),
}])

display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Persist evaluation summary & write HTML report

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {REPORT_SPACE}")

summary_spark = (
    spark.createDataFrame(summary)
    .withColumn("created_at", F.current_timestamp())
)

publish_table(
    spark=spark,
    df=summary_spark,
    table_base=f"{T_ACTIVE_USER_SUMMARY_PREFIX}_{ANALYSIS_NAME}_eval",
    analysis_name=f"{ANALYSIS_NAME}_eval",
    current_mode="view",
    primary_keys=["analysis_name", "run_ts"],
    timeseries_column="run_ts",
)

# Expose overall_mape as a task value so downstream tasks can read it without a table query
dbutils.jobs.taskValues.set(key="overall_mape", value=round(overall_mape, 2))

# COMMAND ----------

# HTML report
generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
status_color = "#16a34a" if overall_mape <= 10 else "#dc2626"
status_label = "PASS" if overall_mape <= 10 else "FAIL"

rows_html = "\n".join(
    f'<tr><td>{str(r["week_end"])[:10]}</td>'
    f'<td style="text-align:right">{r["actual_active"]:,}</td>'
    f'<td style="text-align:right">{r["predicted_active"]:,}</td>'
    f'<td style="text-align:right;color:{"#16a34a" if r["mape"] < 10 else "#dc2626"}">{r["mape"]:.1f}%</td>'
    f'<td style="text-align:right">{r["actual_churn_rate"]*100:.1f}%</td>'
    f'<td style="text-align:right">{r["predicted_churn_rate"]*100:.1f}%</td></tr>'
    for _, r in backtest_df.iterrows()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Active User Prediction – Evaluation – {ANALYSIS_NAME}</title>
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
  <h1>Active User Prediction – Backtest Evaluation</h1>
  <div class="meta">analysis: <strong>{ANALYSIS_NAME}</strong> | alias: {model_alias} | churn threshold: {churn_threshold} | generated: {generated_at}</div>
  <div class="kpi">
    <div class="card"><div class="label">Overall MAPE</div><div class="value" style="color:{status_color}">{overall_mape:.1f}%</div></div>
    <div class="card"><div class="label">Status</div><div class="value" style="color:{status_color}">{status_label}</div></div>
    <div class="card"><div class="label">Threshold</div><div class="value">10%</div></div>
    <div class="card"><div class="label">Folds</div><div class="value">{len(backtest_df)}</div></div>
  </div>
  <table>
    <tr><th>Week End</th><th>Actual Active</th><th>Predicted Active</th><th>MAPE</th><th>Actual Churn Rate</th><th>Pred Churn Rate</th></tr>
    {rows_html}
  </table>
</body>
</html>"""

report_path = f"{REPORT_DIR}/active_user_prediction_report_{ANALYSIS_NAME}.html"
write_info  = write_report_with_single_backup(report_path, html)
print(f"Report written: {write_info['report_path']}")

print(json.dumps({
    "overall_mape":   overall_mape,
    "status":         status_label,
    "num_folds":      len(backtest_df),
}, indent=2))
