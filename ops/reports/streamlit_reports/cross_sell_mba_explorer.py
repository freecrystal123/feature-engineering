from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

REPORT_META = {
    "title": "Cross-sell MBA Explorer",
    "schema": "cross_sell_mba",
    "environments": ("training", "production"),
    "description": "Cross-platform MBA rules with 365d and 30d windows, lift/confidence ordering, and audience estimates.",
}

ANALYSIS_NAMES = {"analysis_4": "ols"}
MBA_ANALYSIS = os.getenv("MBA_ANALYSIS", "analysis_4")
MBA_MIN_BETS_PER_ITEM = 3
MBA_MIN_WAGER_PER_ITEM = 50.0
MBA_MIN_ITEMS_PER_USER = 2
LOOKBACK_WINDOWS = (365, 30)
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")


def _catalog_for_environment(environment: str) -> str:
    env = str(environment).strip().lower()
    if env in {"training", "train", "ml_train"}:
        return "ml_train"
    if env in {"production", "prod", "ml_prod"}:
        return "ml_prod"
    return "ml_prod"


def _environment_key(environment: str) -> str:
    env = str(environment).strip().lower()
    if env in {"training", "train", "ml_train"}:
        return "TRAINING"
    if env in {"production", "prod", "ml_prod"}:
        return "PRODUCTION"
    return "OTHER"


def _schema_for_environment(environment: str, default_schema: str) -> str:
    env_key = _environment_key(environment)
    per_env_override = str(os.getenv(f"MBA_SCHEMA_OVERRIDE_{env_key}", "")).strip()
    global_override = str(os.getenv("MBA_SCHEMA_OVERRIDE", "")).strip()
    return per_env_override or global_override or default_schema


def _catalog_override_for_environment(environment: str, default_catalog: str) -> str:
    env_key = _environment_key(environment)
    per_env_override = str(os.getenv(f"MBA_CATALOG_OVERRIDE_{env_key}", "")).strip()
    global_override = str(os.getenv("MBA_CATALOG_OVERRIDE", "")).strip()
    return per_env_override or global_override or default_catalog


def _analysis_name() -> str:
    override = str(os.getenv("MBA_ANALYSIS_NAME", "")).strip()
    if override:
        return override
    return ANALYSIS_NAMES.get(MBA_ANALYSIS, "ols")


def _table_suffix() -> str:
    # Supported variants:
    # - current  -> *_current
    # - snapshot -> *_snapshot
    # - analysis -> *_<analysis_name> (legacy notebook behavior, e.g. *_ols)
    # Any other non-empty value is used as custom suffix.
    variant = str(os.getenv("MBA_TABLE_VARIANT", "current")).strip().lower()
    if variant == "current":
        return "current"
    if variant == "snapshot":
        return "snapshot"
    if variant == "analysis":
        return _analysis_name()
    return variant or "current"


def _safe_table_name(name: str) -> str:
    table_name = str(name).strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table identifier: {table_name}")
    return table_name


def _resolve_table_names(context: dict) -> tuple[str, str, str]:
    environment = str(context.get("environment", "production")).strip().lower()
    default_schema = str(context.get("schema", "cross_sell_mba")).strip().lower() or "cross_sell_mba"
    schema = _schema_for_environment(environment, default_schema)
    analysis_name = _analysis_name()
    table_suffix = _table_suffix()
    catalog = _catalog_override_for_environment(environment, _catalog_for_environment(environment))

    default_rules_prefix = f"{catalog}.{schema}.mba_association_rules_cross"
    default_transactions_prefix = f"{catalog}.{schema}.mba_transactions"

    rules_prefix = str(os.getenv("MBA_RULES_CROSS_PREFIX", default_rules_prefix)).strip()
    transactions_prefix = str(
        os.getenv("MBA_TRANSACTIONS_PREFIX", default_transactions_prefix)
    ).strip()

    rules_table_override = str(os.getenv("MBA_RULES_TABLE", "")).strip()
    transactions_table_override = str(os.getenv("MBA_TRANSACTIONS_TABLE", "")).strip()

    rules_table = (
        _safe_table_name(rules_table_override)
        if rules_table_override
        else _safe_table_name(f"{rules_prefix}_{table_suffix}")
    )
    transactions_table = (
        _safe_table_name(transactions_table_override)
        if transactions_table_override
        else _safe_table_name(f"{transactions_prefix}_{table_suffix}")
    )
    return rules_table, transactions_table, analysis_name


def _resolve_http_path() -> str:
    for env_name in ("DATABRICKS_SQL_HTTP_PATH", "DATABRICKS_HTTP_PATH"):
        raw = str(os.getenv(env_name, "")).strip()
        if not raw:
            continue

        # If a full HTTP path is provided, use it as-is.
        if raw.startswith("/"):
            return raw
        if raw.startswith("sql/"):
            return f"/{raw}"

        # Databricks Apps `valueFrom: sql_warehouse` resolves to warehouse ID.
        # Convert it to the SQL warehouse HTTP path expected by connector.
        if re.fullmatch(r"[A-Za-z0-9]+", raw):
            return f"/sql/1.0/warehouses/{raw}"

        return raw

    for env_name in ("DATABRICKS_SQL_WAREHOUSE_ID", "WAREHOUSE_ID"):
        warehouse_id = str(os.getenv(env_name, "")).strip()
        if warehouse_id:
            return f"/sql/1.0/warehouses/{warehouse_id}"
    return ""


@st.cache_resource
def _get_connection(http_path: str):
    cfg = Config()
    raw_host = str(cfg.host).strip()
    # Defensively normalize host in case cfg.host contains scheme/path (e.g. /oidc).
    if "://" not in raw_host:
        raw_host = f"https://{raw_host}"
    parsed = urlparse(raw_host)
    host = (parsed.netloc or parsed.path).strip().strip("/")
    if "/" in host:
        host = host.split("/", 1)[0]
    if not host:
        raise ValueError(
            "Unable to resolve Databricks server hostname from Config.host. "
            "Check DATABRICKS_HOST format."
        )
    return sql.connect(
        server_hostname=host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
        _use_arrow_native_complex_types=False,
    )


@st.cache_data(ttl=180)
def _load_rules_dataset(http_path: str, rules_table: str, transactions_table: str) -> pd.DataFrame:
    query = f"""
    WITH rules_base AS (
      SELECT
        r.lookback_days,
        array_sort(r.antecedent) AS antecedent_items,
        COALESCE(r.antecedent_str, concat_ws(' + ', array_sort(r.antecedent))) AS antecedent_str,
        COALESCE(r.consequent_str, concat_ws(' + ', array_sort(r.consequent))) AS consequent_str,
        r.support,
        r.confidence,
        r.lift
      FROM {rules_table} r
      WHERE r.lookback_days IN (30, 365)
    ),
    window_users AS (
      SELECT
        lookback_days,
        COUNT(DISTINCT unique_user_id) AS total_users
      FROM {transactions_table}
      WHERE lookback_days IN (30, 365)
      GROUP BY lookback_days
    ),
    rules_enriched AS (
      SELECT
        b.lookback_days,
        b.antecedent_items,
        b.antecedent_str,
        b.consequent_str,
        b.support,
        b.confidence,
        b.lift,
        CAST(ROUND(b.support * w.total_users) AS BIGINT) AS rule_users,
        CAST(w.total_users AS BIGINT) AS total_users
      FROM rules_base b
      INNER JOIN window_users w
        ON w.lookback_days = b.lookback_days
    ),
    antecedent_users AS (
      SELECT
        r.lookback_days,
        r.antecedent_items,
        COUNT(DISTINCT t.unique_user_id) AS target_audience_users
      FROM (
        SELECT DISTINCT lookback_days, antecedent_items
        FROM rules_enriched
      ) r
      INNER JOIN {transactions_table} t
        ON t.lookback_days = r.lookback_days
       AND size(array_except(r.antecedent_items, t.items)) = 0
      GROUP BY r.lookback_days, r.antecedent_items
    )
    SELECT
      r.lookback_days,
      r.antecedent_str,
      r.consequent_str,
      CAST(r.support AS DOUBLE) AS support,
      CAST(r.confidence AS DOUBLE) AS confidence,
      CAST(r.lift AS DOUBLE) AS lift,
      CAST(r.rule_users AS BIGINT) AS rule_users,
      CAST(COALESCE(a.target_audience_users, 0) AS BIGINT) AS target_audience_users,
      CAST(r.total_users AS BIGINT) AS total_users
    FROM rules_enriched r
    LEFT JOIN antecedent_users a
      ON a.lookback_days = r.lookback_days
     AND a.antecedent_items = r.antecedent_items
    ORDER BY r.lookback_days DESC, r.lift DESC, r.confidence DESC
    """
    conn = _get_connection(http_path)
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall_arrow().to_pandas()


@st.cache_data(ttl=180)
def _load_window_mix(http_path: str, transactions_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      lookback_days,
      CAST(SUM(CASE WHEN EXISTS(items, x -> x LIKE 'sports::%') THEN 1 ELSE 0 END) AS BIGINT) AS sports_users,
      CAST(SUM(CASE WHEN EXISTS(items, x -> x LIKE 'casino::%') THEN 1 ELSE 0 END) AS BIGINT) AS casino_users,
      CAST(
        SUM(
          CASE
            WHEN EXISTS(items, x -> x LIKE 'sports::%')
             AND EXISTS(items, x -> x LIKE 'casino::%') THEN 1
            ELSE 0
          END
        ) AS BIGINT
      ) AS hybrid_users
    FROM {transactions_table}
    WHERE lookback_days IN (30, 365)
    GROUP BY lookback_days
    """
    conn = _get_connection(http_path)
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall_arrow().to_pandas()


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _split_rule_side(side_text: str) -> tuple[str, str]:
    text = str(side_text or "").strip()
    if "::" in text:
        prefix, name = text.split("::", 1)
        return prefix.strip().upper(), name.strip()
    return "ITEM", text


def _render_rule_table(window_df: pd.DataFrame) -> None:
    if window_df.empty:
        st.info("No rules for selected filters.")
        return

    table = window_df.copy()
    table["antecedent"] = table["antecedent_str"]
    table["consequent"] = table["consequent_str"]
    table["support"] = (table["support"] * 100).round(3).astype(str) + "%"
    table["confidence"] = (table["confidence"] * 100).round(2).astype(str) + "%"
    table["lift"] = table["lift"].round(2).astype(str) + "x"
    table["users_in_both"] = table["rule_users"].astype("int64")
    table["target_audience_users"] = table["target_audience_users"].astype("int64")
    st.dataframe(
        table[
            [
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "users_in_both",
                "target_audience_users",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_window_column(days: int, window_df: pd.DataFrame, mix_row: dict) -> None:
    st.markdown(f"#### Last {days} days")
    st.caption("Sorted by lift (desc), then confidence (desc).")

    if window_df.empty:
        st.info("No rules available for this window.")
        return

    total_users = int(window_df["total_users"].iloc[0])
    avg_lift = float(window_df["lift"].mean())
    rules_count = int(len(window_df))
    casino_users = int(mix_row.get("casino_users", 0))
    sports_users = int(mix_row.get("sports_users", 0))
    hybrid_users = int(mix_row.get("hybrid_users", 0))

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Rules", _fmt_int(rules_count))
    metric_col2.metric("Avg Lift", f"{avg_lift:.2f}x")
    metric_col3.metric("Total Users", _fmt_int(total_users))

    mix_col1, mix_col2, mix_col3 = st.columns(3)
    mix_col1.metric("Casino Players", _fmt_int(casino_users))
    mix_col2.metric("Hybrid Players", _fmt_int(hybrid_users))
    mix_col3.metric("Sports Players", _fmt_int(sports_users))

    _render_rule_table(window_df)

    with st.expander("Top rules details", expanded=False):
        for _, row in window_df.head(25).iterrows():
            ant_label, ant_name = _split_rule_side(str(row["antecedent_str"]))
            cons_label, cons_name = _split_rule_side(str(row["consequent_str"]))
            st.markdown(
                (
                    f"**{ant_label}:** {ant_name}  →  **{cons_label}:** {cons_name}\n\n"
                    f"- Lift: `{float(row['lift']):.2f}x`\n"
                    f"- Confidence: `{_fmt_pct(float(row['confidence']))}`\n"
                    f"- Users in Both: `{_fmt_int(int(row['rule_users']))}`\n"
                    f"- Target Audience Users: `{_fmt_int(int(row['target_audience_users']))}`"
                )
            )


def _apply_filters(rules_df: pd.DataFrame, antecedent_filter: str, consequent_filter: str, min_lift: float) -> pd.DataFrame:
    filtered = rules_df.copy()
    if antecedent_filter:
        filtered = filtered[
            filtered["antecedent_str"].str.lower().str.contains(antecedent_filter.lower(), regex=False)
        ]
    if consequent_filter:
        filtered = filtered[
            filtered["consequent_str"].str.lower().str.contains(consequent_filter.lower(), regex=False)
        ]
    filtered = filtered[filtered["lift"] >= float(min_lift)]
    return filtered


def render(context: dict | None = None) -> None:
    context = context or {}
    http_path = _resolve_http_path()
    if not http_path:
        st.error(
            "Missing SQL warehouse HTTP path. Set `DATABRICKS_SQL_HTTP_PATH` "
            "(or `DATABRICKS_HTTP_PATH`) in app environment variables."
        )
        return

    try:
        rules_table, transactions_table, analysis_name = _resolve_table_names(context)
        rules_df = _load_rules_dataset(http_path, rules_table, transactions_table)
        mix_df = _load_window_mix(http_path, transactions_table)
    except Exception as exc:
        st.error(f"Failed to load MBA datasets: {exc}")
        return

    if rules_df.empty:
        st.warning("No rules found for 30d/365d windows.")
        return

    st.subheader(f"MBA Cross-Platform Rules — {analysis_name}")
    st.caption("Rule users are estimated as: support x total users in each time window.")

    badge_col1, badge_col2, badge_col3 = st.columns(3)
    badge_col1.caption(f"Min Bets per Item: {MBA_MIN_BETS_PER_ITEM}")
    badge_col2.caption(f"Min Wager per Item: {MBA_MIN_WAGER_PER_ITEM}")
    badge_col3.caption(f"Min Items per User: {MBA_MIN_ITEMS_PER_USER}")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    antecedent_filter = filter_col1.text_input("Antecedent contains", placeholder="e.g. baccarat")
    consequent_filter = filter_col2.text_input("Consequent contains", placeholder="e.g. football")
    min_lift = filter_col3.number_input("Minimum lift", min_value=0.0, value=0.0, step=0.05)

    filtered = _apply_filters(
        rules_df=rules_df,
        antecedent_filter=antecedent_filter,
        consequent_filter=consequent_filter,
        min_lift=float(min_lift),
    )

    mix_map: dict[int, dict] = {}
    if not mix_df.empty:
        mix_map = {
            int(row["lookback_days"]): {
                "sports_users": int(row["sports_users"]),
                "casino_users": int(row["casino_users"]),
                "hybrid_users": int(row["hybrid_users"]),
            }
            for _, row in mix_df.iterrows()
        }

    left_col, right_col = st.columns(2)
    with left_col:
        df_365 = filtered[filtered["lookback_days"] == LOOKBACK_WINDOWS[0]].copy()
        _render_window_column(LOOKBACK_WINDOWS[0], df_365, mix_map.get(LOOKBACK_WINDOWS[0], {}))

    with right_col:
        df_30 = filtered[filtered["lookback_days"] == LOOKBACK_WINDOWS[1]].copy()
        _render_window_column(LOOKBACK_WINDOWS[1], df_30, mix_map.get(LOOKBACK_WINDOWS[1], {}))
