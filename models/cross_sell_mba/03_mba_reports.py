# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 - MBA Reports
# MAGIC
# MAGIC Goal:
# MAGIC - Build an English HTML report for cross-platform MBA rules.
# MAGIC - Keep the same dark visual theme used in cluster reports.
# MAGIC - Show 365d on the left and 30d on the right.
# MAGIC - Order rules by lift desc, then confidence desc.
# MAGIC - Include estimated users per rule using:
# MAGIC   rule_users = total_users_in_window * support.

# COMMAND ----------

# MAGIC %run ./00_config_cross_sell_mba

# COMMAND ----------

from pyspark.sql import functions as F
from html import escape
from models.utils import read_current_table

# COMMAND ----------

analysis_name = ANALYSIS_NAMES[MBA_ANALYSIS]

rules_df = read_current_table(
    spark=spark,
    table_base=f"{T_MBA_RULES_CROSS_PREFIX}_{analysis_name}",
)
transactions_df = read_current_table(
    spark=spark,
    table_base=f"{T_MBA_TRANSACTIONS_PREFIX}_{analysis_name}",
)

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
    .withColumn("antecedent_items", F.array_sort(F.col("r.antecedent")))
    .withColumn("consequent_items", F.array_sort(F.col("r.consequent")))
    .withColumn("rule_users", F.round(F.col("r.support") * F.col("u.total_users")).cast("long"))
    .withColumn("antecedent_str", F.coalesce(F.col("r.antecedent_str"), F.concat_ws(" + ", F.col("antecedent_items"))))
    .withColumn("consequent_str", F.coalesce(F.col("r.consequent_str"), F.concat_ws(" + ", F.col("consequent_items"))))
    .select(
        "lookback_days",
        "antecedent_items",
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

# Target audience users = users who contain antecedent A in transactions (P(A) * N)
antecedent_users_df = (
    rules_enriched_df
    .select("lookback_days", "antecedent_items")
    .dropDuplicates()
    .alias("ra")
    .join(transactions_df.alias("t"), on="lookback_days", how="inner")
    .filter(F.size(F.array_except(F.col("ra.antecedent_items"), F.col("t.items"))) == 0)
    .groupBy("lookback_days", "antecedent_items")
    .agg(F.countDistinct("t.unique_user_id").cast("long").alias("target_audience_users"))
)

rules_enriched_df = (
    rules_enriched_df.alias("r")
    .join(
        antecedent_users_df.alias("a"),
        on=["lookback_days", "antecedent_items"],
        how="left",
    )
    .withColumn("target_audience_users", F.coalesce(F.col("target_audience_users"), F.lit(0)))
    .orderBy(F.col("lookback_days").desc(), F.col("lift").desc(), F.col("confidence").desc())
)

# COMMAND ----------

rules_365_pdf = (
    rules_enriched_df
    .filter(F.col("lookback_days") == F.lit(365))
    .orderBy(F.col("lift").desc(), F.col("confidence").desc())
    .toPandas()
)
rules_30_pdf = (
    rules_enriched_df
    .filter(F.col("lookback_days") == F.lit(30))
    .orderBy(F.col("lift").desc(), F.col("confidence").desc())
    .toPandas()
)

window_mix_df = (
    transactions_df
    .select(
        "lookback_days",
        "unique_user_id",
        F.expr("exists(items, x -> x like 'sports::%')").cast("int").alias("has_sports"),
        F.expr("exists(items, x -> x like 'casino::%')").cast("int").alias("has_casino"),
    )
    .groupBy("lookback_days")
    .agg(
        F.sum("has_sports").cast("long").alias("sports_users"),
        F.sum("has_casino").cast("long").alias("casino_users"),
        F.sum(F.when((F.col("has_sports") == 1) & (F.col("has_casino") == 1), F.lit(1)).otherwise(F.lit(0))).cast("long").alias("hybrid_users"),
    )
)
window_mix_map = {int(r["lookback_days"]): r.asDict() for r in window_mix_df.collect()}

# COMMAND ----------

def _fmt_pct(x):
    return f"{(float(x) * 100):.2f}%"


def _fmt_num(x):
    return f"{int(x):,}"


def _split_rule_side(side_text: str):
    if "::" in side_text:
        prefix, name = side_text.split("::", 1)
        return prefix.strip().upper(), name.strip()
    return "ITEM", side_text.strip()


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
    mix = window_mix_map.get(int(days), {})
    sports_users = int(mix.get("sports_users", 0))
    casino_users = int(mix.get("casino_users", 0))
    hybrid_users = int(mix.get("hybrid_users", 0))

    rule_cards = []
    for i, row in pdf.reset_index(drop=True).iterrows():
        ant_label, ant_name = _split_rule_side(str(row["antecedent_str"]))
        cons_label, cons_name = _split_rule_side(str(row["consequent_str"]))
        ant_label = escape(ant_label)
        ant_name = escape(ant_name)
        cons_label = escape(cons_label)
        cons_name = escape(cons_name)
        lift = float(row["lift"])
        conf = float(row["confidence"])
        users_both = int(row["rule_users"])
        target_users = int(row["target_audience_users"])
        ant_raw = str(row["antecedent_str"]).lower().replace('"', "&quot;")
        cons_raw = str(row["consequent_str"]).lower().replace('"', "&quot;")

        rule_cards.append(
            f"""
            <li class=\"rule-item\" data-ant="{ant_raw}" data-cons="{cons_raw}" data-lift="{lift:.6f}">
              <div class=\"rule-flow\">
                <div class=\"rule-side\">
                  <div class=\"rule-side-label\">{ant_label}</div>
                  <div class=\"rule-side-name\">{ant_name}</div>
                </div>
                <div class=\"rule-arrow\">&rarr;</div>
                <div class=\"rule-side\">
                  <div class=\"rule-side-label\">{cons_label}</div>
                  <div class=\"rule-side-name\">{cons_name}</div>
                </div>
              </div>
              <ul class=\"metric-bullets\">
                <li><span class=\"material-symbols-rounded metric-icon\">trending_up</span><b>Lift:</b> {lift:.2f}x</li>
                <li><span class=\"material-symbols-rounded metric-icon\">verified</span><b>Confidence:</b> {_fmt_pct(conf)}</li>
                <li><span class=\"material-symbols-rounded metric-icon\">group</span><b>Users in Both:</b> {_fmt_num(users_both)}</li>
                <li><span class=\"material-symbols-rounded metric-icon\">campaign</span><b>Target Audience Users:</b> {_fmt_num(target_users)}</li>
              </ul>
            </li>
            """
        )

    return f"""
    <div class=\"panel\" data-days="{days}">
      <div class=\"panel-title\">Last {days} days</div>
      <div class=\"panel-subtitle\">Sorted by <b>lift</b> (desc), then <b>confidence</b> (desc).</div>

      <div class=\"kpi-head\">
        <div class=\"kpis\">
          <div class=\"kpi\"><div class=\"kpi-label\">Rules</div><div id="rules-count-{days}" class=\"kpi-value\">{_fmt_num(num_rules)}</div></div>
          <div class=\"kpi\"><div class=\"kpi-label\">Avg Lift</div><div id="avg-lift-{days}" class=\"kpi-value\">{avg_lift:.2f}x</div></div>
        </div>

        <div class=\"kpi-subtitle\">Users In Window</div>
        <div class=\"kpi-total\">
          <div class=\"kpi-total-label\">Total Users</div>
          <div class=\"kpi-total-value\">{_fmt_num(total_users)}</div>
        </div>
        <div class=\"kpis\">
          <div class=\"kpi\"><div class=\"kpi-label\">Casino players</div><div class=\"kpi-value\">{_fmt_num(casino_users)}</div></div>
          <div class=\"kpi\"><div class=\"kpi-label\">Hybrid players</div><div class=\"kpi-value\">{_fmt_num(hybrid_users)}</div></div>
          <div class=\"kpi\"><div class=\"kpi-label\">Sports players</div><div class=\"kpi-value\">{_fmt_num(sports_users)}</div></div>
        </div>
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

  <div class=\"thresholds\">
    <span class=\"threshold-title\">Thresholds used in feature eligibility:</span>
    <span class=\"badge\">Min Bets per Item: {MBA_MIN_BETS_PER_ITEM}</span>
    <span class=\"badge\">Min Wager per Item: {MBA_MIN_WAGER_PER_ITEM}</span>
    <span class=\"badge\">Min Items per User: {MBA_MIN_ITEMS_PER_USER}</span>
  </div>
  <div class=\"filters\">
    <div class=\"filter-item\">
      <label for="filter-ant">Antecedent contains</label>
      <input id="filter-ant" type="text" placeholder="e.g. baccarat" />
    </div>
    <div class=\"filter-item\">
      <label for="filter-cons">Consequent contains</label>
      <input id="filter-cons" type="text" placeholder="e.g. football" />
    </div>
    <div class=\"filter-item\">
      <label for="filter-lift">Min Lift</label>
      <input id="filter-lift" type="number" step="0.01" min="0" placeholder="e.g. 2.0" />
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
  --accent: #3BA1F7;
  --accent-soft: rgba(59,161,247,0.22);
}}

.report[data-theme="dark"] {{
  --bg: #0E1117;
  --panel: #161B22;
  --text: #E6EDF3;
  --muted: #9BA3AF;
  --border: rgba(255,255,255,0.08);
  --shadow: rgba(0,0,0,0.45);
  --chip: rgba(255,255,255,0.08);
  --accent: #3BA1F7;
  --accent-soft: rgba(59,161,247,0.22);
}}

.report, .report * {{ box-sizing: border-box; }}
.report {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--accent-soft);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 0 0 1px rgba(59,161,247,0.08) inset;
}}

.header {{
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap: 12px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--accent-soft);
  margin-bottom: 14px;
}}
.title {{ font-size: 22px; font-weight: 800; letter-spacing: 0.2px; line-height: 1.2; }}
.subtitle {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}

.badges {{ display:flex; gap: 8px; flex-wrap: wrap; align-items:center; }}
.badge {{
  font-size: 12px;
  color: var(--muted);
  border: 1px solid var(--accent-soft);
  background: rgba(59,161,247,0.10);
  padding: 6px 10px;
  border-radius: 999px;
  white-space: nowrap;
}}
.thresholds {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}}
.threshold-title {{
  color: var(--muted);
  font-size: 13px;
}}
.filters {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}}
.filter-item {{
  border: 1px solid var(--border);
  border-radius: 12px;
  background: rgba(59,161,247,0.08);
  padding: 10px;
}}
.filter-item label {{
  display: block;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 6px;
}}
.filter-item input {{
  width: 100%;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 13px;
  outline: none;
}}
.filter-item input:focus {{
  border-color: rgba(255,255,255,0.28);
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
.kpi-head {{
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 10px;
  margin-bottom: 12px;
}}

.kpis {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}}
.kpi {{
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 10px;
}}
.kpi-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
.kpi-value {{ font-size: 16px; font-weight: 900; }}
.kpi-subtitle {{ font-size: 12px; color: var(--text); margin-bottom: 8px; font-weight: 700; }}
.kpi-total {{
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.05);
  border-radius: 12px;
  padding: 10px;
  margin-bottom: 10px;
  text-align: center;
}}
.kpi-total-label {{ font-size: 12px; color: var(--text); margin-bottom: 4px; font-weight: 700; }}
.kpi-total-value {{ font-size: 24px; font-weight: 900; }}

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
.rule-flow {{
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: stretch;
  gap: 10px;
  margin-bottom: 10px;
}}
.rule-side {{
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px;
  background: rgba(255,255,255,0.03);
}}
.rule-side-label {{
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .5px;
  color: var(--muted);
  margin-bottom: 4px;
}}
.rule-side-name {{ font-size: 14px; font-weight: 700; line-height: 1.25; }}
.rule-arrow {{
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  font-weight: 900;
  color: var(--accent);
}}
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
  color: var(--accent);
}}

.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}

@media (max-width: 1200px) {{
  .twocol {{ grid-template-columns: 1fr; }}
  .filters {{ grid-template-columns: 1fr; }}
}}

.displayedHTML,
.displayedHTML > div,
.displayedHTML .output_html,
.displayedHTML .output_subarea,
body {{
  background: #000 !important;
}}
</style>

<script>
(() => {{
  const antInput = document.getElementById('filter-ant');
  const consInput = document.getElementById('filter-cons');
  const liftInput = document.getElementById('filter-lift');
  const panels = Array.from(document.querySelectorAll('.panel[data-days]'));

  const fmtInt = (n) => new Intl.NumberFormat('en-US').format(n);
  const fmtLift = (n) => `${{n.toFixed(2)}}x`;

  function applyFilters() {{
    const ant = (antInput.value || '').trim().toLowerCase();
    const cons = (consInput.value || '').trim().toLowerCase();
    const minLift = parseFloat(liftInput.value || '0');

    panels.forEach((panel) => {{
      const days = panel.getAttribute('data-days');
      const cards = Array.from(panel.querySelectorAll('.rule-item'));
      let visible = 0;
      let liftSum = 0;

      cards.forEach((card) => {{
        const a = card.getAttribute('data-ant') || '';
        const c = card.getAttribute('data-cons') || '';
        const l = parseFloat(card.getAttribute('data-lift') || '0');
        const ok = (!ant || a.includes(ant)) && (!cons || c.includes(cons)) && (l >= minLift);
        card.style.display = ok ? '' : 'none';
        if (ok) {{
          visible += 1;
          liftSum += l;
        }}
      }});

      const rulesEl = document.getElementById(`rules-count-${{days}}`);
      const avgLiftEl = document.getElementById(`avg-lift-${{days}}`);
      if (rulesEl) rulesEl.textContent = fmtInt(visible);
      if (avgLiftEl) avgLiftEl.textContent = visible > 0 ? fmtLift(liftSum / visible) : '0.00x';
    }});
  }}

  [antInput, consInput, liftInput].forEach(el => el && el.addEventListener('input', applyFilters));
}})();
</script>
"""

# COMMAND ----------

report_path = f"{REPORT_DIR}/mba_rules_report_{analysis_name}.html"
with open(report_path, "w") as f:
    f.write(html)

print(f"Report saved into {report_path}")

# COMMAND ----------

displayHTML(html)
