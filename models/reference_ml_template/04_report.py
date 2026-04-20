# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 - Reference HTML report

# COMMAND ----------

# MAGIC %run ./00_config_reference_ml_template

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from datetime import datetime, timezone

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
analysis_id = dbutils.widgets.get("analysis_id").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


spark.sql(f"CREATE VOLUME IF NOT EXISTS {REPORT_SPACE}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    feat_df = read_current_table(
        spark=spark,
        table_base=f"{T_REFERENCE_FEATURES_PREFIX}_{analysis_name}",
    )
    pred_df = read_current_table(
        spark=spark,
        table_base=f"{T_REFERENCE_PREDICTIONS_PREFIX}_{analysis_name}",
    )

    joined = feat_df.alias("f").join(
        pred_df.alias("p").select("unique_user_id", "predicted_label"),
        on="unique_user_id",
        how="left",
    )

    summary = joined.agg(
        F.count("*").alias("num_users"),
        F.avg(F.col("f.label_active_7d").cast("double")).alias("actual_rate"),
        F.avg(F.col("predicted_label").cast("double")).alias("predicted_rate"),
        F.avg(F.col("f.num_bets_90d").cast("double")).alias("avg_num_bets_90d"),
        F.avg(F.col("f.total_stake_90d").cast("double")).alias("avg_total_stake_90d"),
    ).collect()[0]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Reference ML Report - {analysis_name}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --ink: #1f2a37;
      --card: #ffffff;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: radial-gradient(circle at top right, #d1fae5 0%, var(--bg) 45%);
      color: var(--ink);
      font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ opacity: 0.8; margin-bottom: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; }}
    .value {{ font-size: 28px; font-weight: 700; color: var(--accent); margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Reference ML Template Report</h1>
  <div class=\"meta\">analysis: <strong>{analysis_name}</strong> | generated at: {generated_at}</div>

  <div class=\"grid\">
    <div class=\"card\"><div class=\"label\">Users scored</div><div class=\"value\">{int(summary['num_users'] or 0):,}</div></div>
    <div class=\"card\"><div class=\"label\">Actual active 7d rate</div><div class=\"value\">{float(summary['actual_rate'] or 0.0):.3f}</div></div>
    <div class=\"card\"><div class=\"label\">Predicted positive rate</div><div class=\"value\">{float(summary['predicted_rate'] or 0.0):.3f}</div></div>
    <div class=\"card\"><div class=\"label\">Avg bets (90d)</div><div class=\"value\">{float(summary['avg_num_bets_90d'] or 0.0):.2f}</div></div>
    <div class=\"card\"><div class=\"label\">Avg total stake (90d)</div><div class=\"value\">{float(summary['avg_total_stake_90d'] or 0.0):.2f}</div></div>
  </div>
</body>
</html>
""".strip()

    report_path = f"{REPORT_DIR}/reference_report_{analysis_name}.html"
    write_info = write_report_with_single_backup(report_path, html)
    print(f"Report written: {write_info['report_path']}")
