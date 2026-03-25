# Databricks notebook source
# MAGIC %md
# MAGIC ## 5.2 - MBA Reports
# MAGIC
# MAGIC Goal:
# MAGIC - Build an English HTML report for cross-platform MBA rules.
# MAGIC - Keep the same dark visual theme used in cluster reports.
# MAGIC - Show 365d on the left and 30d on the right.
# MAGIC - Order rules by lift desc, then confidence desc.
# MAGIC - Include estimated users per rule using:
# MAGIC   rule_users = total_users_in_window * support.

# COMMAND ----------

# MAGIC %run ./050_config_mba

# COMMAND ----------

from pyspark.sql import functions as F
from html import escape

# COMMAND ----------

analysis_name = ANALYSIS_NAMES[MBA_ANALYSIS]

rules_df = spark.table(f"{T_MBA_RULES_CROSS_PREFIX}_{analysis_name}")
transactions_df = spark.table(f"{T_MBA_TRANSACTIONS_PREFIX}_{analysis_name}")

# Ensure report volume exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS {REPORT_SPACE}")

# COMMAND ----------

window_users_df = (
    transactions_df
    .groupBy("lookback_days")
    .agg(F.countDistinct("unique_user_id").alias("total_users"))
)

rules_enriched_df = (
    rules_df.alias("r")
    .join(window_users_df.alias("u"), on="lookback_days", how="inner")
    .withColumn("rule_users", F.round(F.col("r.support") * F.col("u.total_users")).cast("long"))
    .withColumn("antecedent_str", F.coalesce(F.col("r.antecedent_str"), F.concat_ws(" + ", F.col("r.antecedent"))))
    .withColumn("consequent_str", F.coalesce(F.col("r.consequent_str"), F.concat_ws(" + ", F.col("r.consequent"))))
    .select(
        "lookback_days",
        "antecedent_str",
        "consequent_str",
        "support",
        "confidence",
        "lift",
        "rule_users",
        "total_users",
    )
    .orderBy(F.col("lookback_days").desc(), F.col("lift").desc(), F.col("confidence").desc())
)

# COMMAND ----------

rules_365_pdf = rules_enriched_df.filter(F.col("lookback_days") == F.lit(365)).toPandas()
rules_30_pdf = rules_enriched_df.filter(F.col("lookback_days") == F.lit(30)).toPandas()

# COMMAND ----------

def _fmt_pct(x):
    return f"{(float(x) * 100):.2f}%"


def _fmt_num(x):
    return f"{int(x):,}"


def _render_window_column(days: int, pdf):
    if pdf.empty:
        return f"""
        <div class=\"panel\">
          <div class=\"panel-title\">Last {days} days</div>
          <div class=\"panel-subtitle\">No rules available for this window.</div>
        </div>
        """

    total_users = int(pdf["total_users"].iloc[0])
    num_rules = int(len(pdf))
    avg_lift = float(pdf["lift"].mean())

    rule_cards = []
    for i, row in pdf.reset_index(drop=True).iterrows():
        ant = escape(str(row["antecedent_str"]))
        cons = escape(str(row["consequent_str"]))
        lift = float(row["lift"])
        conf = float(row["confidence"])
        supp = float(row["support"])
        users = int(row["rule_users"])
        conf_class = " confidence-high" if conf >= 0.45 else ""

        rule_cards.append(
            f"""
            <li class=\"rule-item{conf_class}\">
              <div class=\"rule-title\">{ant} &rarr; {cons}</div>
              <ul class=\"metric-bullets\">
                <li><span class=\"material-symbols-rounded metric-icon\">trending_up</span><b>Lift:</b> {lift:.2f}x</li>
                <li><span class=\"material-symbols-rounded metric-icon\">verified</span><b>Confidence:</b> {_fmt_pct(conf)}</li>
                <li><span class=\"material-symbols-rounded metric-icon\">group</span><b>Estimated users:</b> {_fmt_num(users)}</li>
              </ul>
            </li>
            """
        )

    return f"""
    <div class=\"panel\">
      <div class=\"panel-title\">Last {days} days</div>
      <div class=\"panel-subtitle\">Sorted by <b>lift</b> (desc), then <b>confidence</b> (desc).</div>

      <div class=\"kpis\">
        <div class=\"kpi\"><div class=\"kpi-label\">Rules</div><div class=\"kpi-value\">{_fmt_num(num_rules)}</div></div>
        <div class=\"kpi\"><div class=\"kpi-label\">Users in window</div><div class=\"kpi-value\">{_fmt_num(total_users)}</div></div>
        <div class=\"kpi\"><div class=\"kpi-label\">Avg Lift</div><div class=\"kpi-value\">{avg_lift:.2f}x</div></div>
      </div>

      <ol class=\"rule-list\">{''.join(rule_cards)}</ol>
    </div>
    """


left_col_html = _render_window_column(365, rules_365_pdf)
right_col_html = _render_window_column(30, rules_30_pdf)

html = f"""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20,600,1,0" />
<div class=\"report\" data-theme=\"dark\">
  <div class=\"header\">
    <div>
      <div class=\"title\">MBA Cross-Platform Rules — {analysis_name}</div>
      <div class=\"subtitle\">Rule users are estimated as: <span class=\"mono\">support x total users in each time window</span>.</div>
    </div>
    <div class=\"badges\">
      <span class=\"badge\">Windows: 365d and 30d</span>
      <span class=\"badge\">Scope: Cross-platform rules</span>
    </div>
  </div>

  <div class=\"twocol\">
    {left_col_html}
    {right_col_html}
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

.twocol {{
  display: grid;
  grid-template-columns: 1fr 1fr;
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

.panel-title {{ font-weight: 800; font-size: 14px; }}
.panel-subtitle {{ font-size: 12px; color: var(--muted); margin-top: 4px; margin-bottom: 10px; line-height: 1.35; }}

.kpis {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}}
.kpi {{
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 10px;
}}
.kpi-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
.kpi-value {{ font-size: 16px; font-weight: 900; }}

.rule-list {{
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: 10px;
}}
.rule-item {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255,255,255,0.02);
}}
.rule-title {{ font-weight: 700; margin-bottom: 8px; line-height: 1.3; }}
.metric-bullets {{
  margin: 0;
  padding-left: 0;
  color: var(--text);
  line-height: 1.4;
  list-style: none;
}}
.metric-bullets li {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.metric-icon {{
  font-size: 18px;
  line-height: 1;
  color: var(--muted);
}}
.rule-item.confidence-high,
.rule-item.confidence-high .rule-title,
.rule-item.confidence-high .metric-bullets,
.rule-item.confidence-high .metric-bullets b,
.rule-item.confidence-high .metric-icon {{
  color: #00D18F !important;
}}

.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}

@media (max-width: 1200px) {{
  .twocol {{ grid-template-columns: 1fr; }}
}}

.displayedHTML,
.displayedHTML > div,
.displayedHTML .output_html,
.displayedHTML .output_subarea,
body {{
  background: #000 !important;
}}
</style>
"""

# COMMAND ----------

report_path = f"{REPORT_DIR}/mba_rules_report_{analysis_name}.html"
with open(report_path, "w") as f:
    f.write(html)

print(f"Report saved into {report_path}")

# COMMAND ----------

displayHTML(html)
