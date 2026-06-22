from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select, resolve_sql_http_path
from reports._funnel_metadata import load_report_freshness, render_report_freshness

REPORT_META = {
    "title": "Deposit-to-First-Wager Journey",
    "schema": "conversion_funnel_bet",
    "environments": ("training",),
    "description": "Operational journey from first successful deposit to first valid wager.",
}

TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
ANALYSIS_NAMES = {
    "analysis_1": "deposit_to_wager_30d",
    "analysis_2": "deposit_to_wager_30d_truewin",
    "analysis_3": "deposit_to_wager_30d_igaming",
    "analysis_4": "deposit_to_wager_30d_truewin_no_ekyc",
    "analysis_5": "deposit_to_wager_30d_truewin_no_ekyc_no_remarks_coupon",
}


def _analysis_name() -> str:
    analysis_id = os.getenv("BET_FUNNEL_JOURNEY_ANALYSIS", "analysis_2")
    return ANALYSIS_NAMES.get(analysis_id, "deposit_to_wager_30d_truewin")


def _analysis_options() -> list[str]:
    return list(dict.fromkeys(ANALYSIS_NAMES[key] for key in sorted(ANALYSIS_NAMES)))


def _environment_key(environment: str) -> str:
    return "TRAINING" if str(environment).lower() in {"training", "train", "ml_train"} else "PRODUCTION"


def _safe_table_name(name: str) -> str:
    table_name = str(name).strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table identifier: {table_name}")
    return table_name


def _resolve_table_name(context: dict, analysis_name: str) -> str:
    env_key = _environment_key(str(context.get("environment", "production")))
    catalog = "ml_train" if env_key == "TRAINING" else "ml_prod"
    default = f"{catalog}.conversion_funnel_bet.rpt_deposit_journey_{analysis_name}_current"
    override = str(os.getenv(f"BET_FUNNEL_JOURNEY_TABLE_{env_key}", "")).strip()
    return _safe_table_name(override or default)


def _matured_where(matured_only: bool) -> str:
    return "WHERE observation_complete = TRUE" if matured_only else ""


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_summary(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT
          COUNT(*) AS depositors,
          SUM(CAST(label_wager_success_30d AS BIGINT)) AS wagered_30d,
          AVG(CAST(label_wager_success_30d AS DOUBLE)) AS conversion_rate_30d,
          AVG(CASE WHEN label_wager_success_30d = 1 THEN first_wager_delay_hours_30d END) AS avg_hours_to_wager,
          percentile_approx(CASE WHEN label_wager_success_30d = 1 THEN first_wager_delay_hours_30d END, 0.5) AS median_hours_to_wager,
          AVG(first_deposit_amount) AS avg_first_deposit_amount,
          AVG(num_wagers_30d) AS avg_wagers_30d,
          SUM(sum_wager_amount_30d) AS total_wager_amount_30d,
          MAX(journey_window_days) AS journey_window_days
        FROM {table_name}
        {_matured_where(matured_only)}
        """,
        http_path=http_path,
    )


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_trend(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT CAST(deposit_month AS DATE) AS deposit_month,
               COUNT(*) AS depositors,
               AVG(CAST(label_wager_success_30d AS DOUBLE)) AS conversion_rate_30d,
               percentile_approx(
                 CASE WHEN label_wager_success_30d = 1 THEN first_wager_delay_hours_30d END,
                 0.5
               ) AS median_hours_to_wager
        FROM {table_name}
        {_matured_where(matured_only)}
        GROUP BY deposit_month
        ORDER BY deposit_month
        """,
        http_path=http_path,
    )


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_delay_buckets(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT time_to_first_wager_bucket AS bucket,
               COUNT(*) AS users,
               AVG(CAST(label_wager_success_30d AS DOUBLE)) AS conversion_rate
        FROM {table_name}
        {_matured_where(matured_only)}
        GROUP BY time_to_first_wager_bucket
        """,
        http_path=http_path,
    )


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_amount_buckets(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT deposit_amount_bucket AS bucket,
               COUNT(*) AS depositors,
               AVG(CAST(label_wager_success_30d AS DOUBLE)) AS conversion_rate,
               AVG(CASE WHEN label_wager_success_30d = 1 THEN first_wager_delay_hours_30d END) AS avg_hours_to_wager
        FROM {table_name}
        {_matured_where(matured_only)}
        GROUP BY deposit_amount_bucket
        """,
        http_path=http_path,
    )


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_sources(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT source_system,
               COUNT(*) AS depositors,
               AVG(CAST(label_wager_success_30d AS DOUBLE)) AS conversion_rate,
               percentile_approx(
                 CASE WHEN label_wager_success_30d = 1 THEN first_wager_delay_hours_30d END,
                 0.5
               ) AS median_hours_to_wager
        FROM {table_name}
        {_matured_where(matured_only)}
        GROUP BY source_system
        ORDER BY depositors DESC
        """,
        http_path=http_path,
    )


def _fmt_pct(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):.1%}"


def _fmt_num(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{int(value):,}"


def _fmt_hours(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):,.1f} h"


def get_last_updated(context: dict | None = None):
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        return "-"
    table_name = _resolve_table_name(context, _analysis_name())
    return load_report_freshness(http_path, table_name).get("report_updated_at") or "-"


def render(context: dict | None = None) -> None:
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        st.error("Missing SQL warehouse HTTP path.")
        return

    options = _analysis_options()
    analysis_name = st.selectbox(
        "Analysis",
        options=options,
        index=options.index(_analysis_name()) if _analysis_name() in options else 0,
        key=f"funnel_bet_journey_analysis_{str(context.get('environment', 'other')).lower()}",
    )
    matured_only = st.checkbox(
        "Matured cohorts only",
        value=True,
        key=f"funnel_bet_journey_matured_{str(context.get('environment', 'other')).lower()}",
    )

    try:
        table_name = _resolve_table_name(context, analysis_name)
        summary_df = _load_summary(http_path, table_name, matured_only)
        trend_df = _load_trend(http_path, table_name, matured_only)
        delay_df = _load_delay_buckets(http_path, table_name, matured_only)
        amount_df = _load_amount_buckets(http_path, table_name, matured_only)
        source_df = _load_sources(http_path, table_name, matured_only)
    except Exception as exc:
        st.error(f"Failed to load deposit-to-wager journey: {exc}")
        return

    if summary_df.empty:
        st.warning("No journey data found.")
        return
    row = summary_df.iloc[0]

    st.subheader(f"Deposit -> First Wager Journey ({analysis_name})")
    st.caption("The clock starts at the first successful deposit. A conversion is the first valid wager placement within 30 days.")
    try:
        render_report_freshness(load_report_freshness(http_path, table_name))
    except Exception:
        pass

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Depositors", _fmt_num(row.get("depositors")))
    k2.metric("Wager conversion", _fmt_pct(row.get("conversion_rate_30d")))
    k3.metric("Median time to wager", _fmt_hours(row.get("median_hours_to_wager")))
    k4.metric("Avg wagers in 30d", "-" if pd.isna(row.get("avg_wagers_30d")) else f"{float(row.get('avg_wagers_30d')):,.1f}")

    st.markdown("##### Monthly activation trend")
    if not trend_df.empty:
        trend_df["deposit_month"] = pd.to_datetime(trend_df["deposit_month"], errors="coerce")
        st.altair_chart(
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("deposit_month:T", title="First-deposit month"),
                y=alt.Y("conversion_rate_30d:Q", title="Wager conversion", axis=alt.Axis(format="%")),
                tooltip=["deposit_month:T", "depositors:Q", alt.Tooltip("conversion_rate_30d:Q", format=".1%")],
            )
            .properties(height=300),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("##### Time to first wager")
        if not delay_df.empty:
            order = ["< 1 hour", "1-6 hours", "6-24 hours", "1-3 days", "3-7 days", "7-30 days", "No wager within 30d"]
            st.altair_chart(
                alt.Chart(delay_df)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", sort=order, title=None),
                    y=alt.Y("users:Q", title="Users"),
                    color=alt.Color("bucket:N", legend=None),
                    tooltip=["bucket:N", "users:Q"],
                )
                .properties(height=320),
                use_container_width=True,
            )
    with right:
        st.markdown("##### Conversion by first-deposit amount band")
        if not amount_df.empty:
            amount_order = ["<= 10", "10-25", "25-50", "50-100", "> 100", "Missing"]
            st.altair_chart(
                alt.Chart(amount_df)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", sort=amount_order, title="Amount band (native currency)"),
                    y=alt.Y("conversion_rate:Q", title="Conversion", axis=alt.Axis(format="%")),
                    tooltip=["bucket:N", "depositors:Q", alt.Tooltip("conversion_rate:Q", format=".1%")],
                )
                .properties(height=320),
                use_container_width=True,
            )

    st.markdown("##### Source-system comparison")
    if not source_df.empty:
        show = source_df.copy()
        show["conversion_rate"] = show["conversion_rate"].apply(_fmt_pct)
        show["median_hours_to_wager"] = show["median_hours_to_wager"].apply(_fmt_hours)
        st.dataframe(show, use_container_width=True, hide_index=True)
