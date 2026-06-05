from __future__ import annotations

import os
import re
import html

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select, resolve_sql_http_path
from reports._funnel_metadata import load_report_freshness, render_report_freshness

REPORT_META = {
    "title": "Deposit Conversion Funnel eKYC-to-FTD Journey",
    "schema": "conversion_funnel_deposit",
    "environments": ("training",),
    "description": "eKYC-to-FTD customer journey from registration through explicit verification statuses and first-time deposit outcomes.",
}

FUNNEL_ANALYSIS = os.getenv("FUNNEL_ANALYSIS", "analysis_2")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")

ANALYSIS_NAMES = {
    "analysis_1": "register_to_deposit_30d",
    "analysis_2": "register_to_deposit_30d_truewin",
    "analysis_3": "register_to_deposit_30d_igaming",
    "analysis_4": "register_to_deposit_30d_truewin_no_ekyc",
}

STATUS_SIGNALS = [
    ("eKYC under review", "COALESCE(has_ekyc_under_review_30d, 0) = 1"),
    ("eKYC explicit failure", "COALESCE(has_ekyc_failure_30d, 0) = 1"),
    ("eKYC expired", "COALESCE(has_ekyc_expired_30d, 0) = 1"),
    ("eKYC black list", "COALESCE(has_ekyc_black_list_30d, 0) = 1"),
    ("Recovered after eKYC failure", "COALESCE(ekyc_recovered_after_failure, 0) = 1"),
    ("Verified after under review", "COALESCE(ekyc_recovered_after_review, 0) = 1"),
    ("FTD failure, not recovered", "COALESCE(has_ftd_failure_unrecovered, 0) = 1"),
    ("FTD pending / handling unresolved", "COALESCE(has_ftd_pending_unresolved, 0) = 1"),
]


def _environment_key(environment: str) -> str:
    env = str(environment).strip().lower()
    if env in {"training", "train", "ml_train"}:
        return "TRAINING"
    if env in {"production", "prod", "ml_prod"}:
        return "PRODUCTION"
    return "OTHER"


def _analysis_name() -> str:
    override = str(os.getenv("FUNNEL_ANALYSIS_NAME", "")).strip()
    if override:
        return override
    return ANALYSIS_NAMES.get(FUNNEL_ANALYSIS, "register_to_deposit_30d_truewin")


def _analysis_options() -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for _, name in sorted(ANALYSIS_NAMES.items()):
        if name not in seen:
            options.append(name)
            seen.add(name)
    return options


def _safe_table_name(name: str) -> str:
    table_name = str(name).strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table identifier: {table_name}")
    return table_name


def _resolve_table_name(context: dict, analysis_name: str) -> str:
    environment = str(context.get("environment", "production")).strip().lower()
    env_key = _environment_key(environment)
    default_catalog = "ml_train" if env_key == "TRAINING" else "ml_prod"
    default_table = f"{default_catalog}.conversion_funnel_deposit.rpt_ekyc_journey_{analysis_name}_current"
    override = (
        str(os.getenv(f"FUNNEL_EKYC_JOURNEY_TABLE_{env_key}", "")).strip()
        or str(os.getenv("FUNNEL_EKYC_JOURNEY_TABLE", "")).strip()
    )
    return _safe_table_name(override) if override else _safe_table_name(default_table)


def _where_clause(matured_only: bool) -> str:
    return "WHERE COALESCE(observation_complete, FALSE) = TRUE" if matured_only else ""


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def _fmt_hours(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    hours = float(value)
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


@st.cache_data(ttl=180)
def _load_summary(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    SELECT
      COUNT(*) AS users,
      MAX(CAST(journey_window_days AS INT)) AS journey_window_days,
      MAX(verification_history_source) AS verification_history_source,
      MAX(CAST(COALESCE(coupon_source_available, FALSE) AS INT)) AS coupon_source_available,
      AVG(CAST(COALESCE(has_ekyc_event_30d, 0) AS DOUBLE)) AS ekyc_start_rate,
      AVG(CAST(COALESCE(has_ekyc_verified_30d, 0) AS DOUBLE)) AS ekyc_verified_rate,
      AVG(CAST(COALESCE(has_ekyc_failure_30d, 0) AS DOUBLE)) AS explicit_failure_rate,
      AVG(CAST(COALESCE(has_ekyc_under_review_30d, 0) AS DOUBLE)) AS under_review_rate,
      AVG(CAST(COALESCE(has_ftd_attempt_30d, 0) AS DOUBLE)) AS ftd_attempt_rate,
      AVG(CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE)) AS ftd_conversion_rate,
      AVG(CAST(COALESCE(has_coupon_before_first_ftd, 0) AS DOUBLE)) AS coupon_before_ftd_rate,
      AVG(CASE WHEN COALESCE(has_ekyc_failure_30d, 0) = 1 THEN CAST(COALESCE(ekyc_recovered_after_failure, 0) AS DOUBLE) END) AS recovery_after_failure_rate,
      AVG(CASE WHEN COALESCE(has_ekyc_under_review_30d, 0) = 1 THEN CAST(COALESCE(ekyc_recovered_after_review, 0) AS DOUBLE) END) AS review_to_verified_rate,
      AVG(CASE WHEN COALESCE(has_ekyc_verified_30d, 0) = 1 THEN CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE) END) AS conversion_after_verified_rate,
      AVG(CASE WHEN COALESCE(has_ftd_attempt_30d, 0) = 1 THEN CAST(COALESCE(has_ekyc_before_first_ftd_attempt, 0) AS DOUBLE) END) AS ekyc_before_ftd_attempt_rate,
      AVG(ekyc_recovery_hours_after_failure) AS avg_ekyc_recovery_hours,
      AVG(ekyc_review_to_verified_hours) AS avg_review_to_verified_hours,
      AVG(hours_ekyc_verified_to_first_ftd_attempt) AS avg_verified_to_ftd_attempt_hours
    FROM {table_name}
    {_where_clause(matured_only)}
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_funnel_stages(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    where_clause = _where_clause(matured_only)
    query = f"""
    WITH base AS (
      SELECT *
      FROM {table_name}
      {where_clause}
    ),
    stages AS (
      SELECT 1 AS stage_order, 'Registered' AS stage, COUNT(*) AS users FROM base
      UNION ALL
      SELECT 2, 'eKYC started', SUM(COALESCE(has_ekyc_event_30d, 0)) FROM base
      UNION ALL
      SELECT 3, 'eKYC verified', SUM(COALESCE(has_ekyc_verified_30d, 0)) FROM base
      UNION ALL
      SELECT 4, 'FTD attempted', SUM(COALESCE(has_ftd_attempt_30d, 0)) FROM base
      UNION ALL
      SELECT 5, 'FTD converted', SUM(COALESCE(has_ftd_success_30d, 0)) FROM base
    )
    SELECT
      stage_order,
      stage,
      CAST(users AS BIGINT) AS users,
      CAST(users AS DOUBLE) / NULLIF(MAX(CASE WHEN stage_order = 1 THEN users END) OVER (), 0) AS pct_registered
    FROM stages
    ORDER BY stage_order
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_monthly_trend(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    SELECT
      CAST(registration_month AS DATE) AS registration_month,
      COUNT(*) AS users,
      AVG(CAST(COALESCE(has_ekyc_event_30d, 0) AS DOUBLE)) AS ekyc_start_rate,
      AVG(CAST(COALESCE(has_ekyc_verified_30d, 0) AS DOUBLE)) AS ekyc_verified_rate,
      AVG(CAST(COALESCE(has_ekyc_failure_30d, 0) AS DOUBLE)) AS explicit_failure_rate,
      AVG(CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE)) AS ftd_conversion_rate
    FROM {table_name}
    {_where_clause(matured_only)}
    GROUP BY registration_month
    ORDER BY registration_month
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_status_signals(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    where_clause = _where_clause(matured_only)
    select_parts = []
    for signal, condition in STATUS_SIGNALS:
        safe_signal = signal.replace("'", "''")
        select_parts.append(
            f"""
            SELECT
              '{safe_signal}' AS signal,
              SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) AS users,
              AVG(CASE WHEN {condition} THEN CAST(COALESCE(has_ftd_attempt_30d, 0) AS DOUBLE) END) AS ftd_attempt_rate,
              AVG(CASE WHEN {condition} THEN CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE) END) AS ftd_conversion_rate
            FROM base
            """
        )
    query = f"""
    WITH base AS (
      SELECT *
      FROM {table_name}
      {where_clause}
    ),
    total AS (
      SELECT COUNT(*) AS users FROM base
    ),
    signals AS (
      {" UNION ALL ".join(select_parts)}
    )
    SELECT
      s.signal,
      CAST(s.users AS BIGINT) AS users,
      CAST(s.users AS DOUBLE) / NULLIF(t.users, 0) AS population_rate,
      s.ftd_attempt_rate,
      s.ftd_conversion_rate
    FROM signals s
    CROSS JOIN total t
    WHERE s.users > 0
    ORDER BY s.users DESC, s.signal
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_journey_outcomes(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    WITH grouped AS (
      SELECT
        journey_outcome,
        COUNT(*) AS users,
        AVG(CAST(COALESCE(has_coupon_before_first_ftd, 0) AS DOUBLE)) AS coupon_rate
      FROM {table_name}
      {_where_clause(matured_only)}
      GROUP BY journey_outcome
    )
    SELECT
      journey_outcome,
      CAST(users AS BIGINT) AS users,
      CAST(users AS DOUBLE) / NULLIF(SUM(users) OVER (), 0) AS population_rate,
      coupon_rate
    FROM grouped
    ORDER BY users DESC, journey_outcome
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_flow_paths(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    where_clause = _where_clause(matured_only)
    query = f"""
    WITH base AS (
      SELECT *
      FROM {table_name}
      {where_clause}
    ),
    classified AS (
      SELECT
        CASE
          WHEN COALESCE(has_ftd_success_30d, 0) = 1
            AND COALESCE(ekyc_recovered_after_failure, 0) = 1
            THEN 'recovered_converted'
          WHEN COALESCE(has_ftd_success_30d, 0) = 1
            AND COALESCE(has_ekyc_failure_30d, 0) = 1
            THEN 'critical_converted'
          WHEN COALESCE(has_ftd_success_30d, 0) = 1
            AND COALESCE(has_ekyc_verified_30d, 0) = 1
            THEN 'healthy_converted'
          WHEN COALESCE(has_ftd_success_30d, 0) = 1
            THEN 'converted_without_observed_approval'
          WHEN COALESCE(ekyc_recovered_after_failure, 0) = 1
            THEN 'recovered_no_ftd'
          WHEN COALESCE(has_ekyc_failure_30d, 0) = 1
            THEN 'failure_no_ftd'
          WHEN COALESCE(has_ekyc_verified_30d, 0) = 1
            THEN 'verified_no_ftd'
          WHEN COALESCE(has_ekyc_event_30d, 0) = 1
            THEN 'started_no_ftd'
          ELSE 'no_ekyc_no_ftd'
        END AS flow_key
      FROM base
    ),
    grouped AS (
      SELECT flow_key, COUNT(*) AS users
      FROM classified
      GROUP BY flow_key
    )
    SELECT
      flow_key,
      CAST(users AS BIGINT) AS users,
      CAST(users AS DOUBLE) / NULLIF(SUM(users) OVER (), 0) AS population_rate
    FROM grouped
    WHERE users > 0
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_delay_buckets(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    where_clause = _where_clause(matured_only)
    query = f"""
    WITH base AS (
      SELECT *
      FROM {table_name}
      {where_clause}
    ),
    events AS (
      SELECT 'First eKYC event' AS process, first_ekyc_event_delay_hours_30d AS delay_hours, has_ftd_success_30d AS converted FROM base
      UNION ALL
      SELECT 'First under review', first_ekyc_under_review_delay_hours_30d, has_ftd_success_30d FROM base
      UNION ALL
      SELECT 'First explicit failure', first_ekyc_failure_delay_hours_30d, has_ftd_success_30d FROM base
      UNION ALL
      SELECT 'First eKYC verified', first_ekyc_verified_delay_hours_30d, has_ftd_success_30d FROM base
      UNION ALL
      SELECT 'First FTD attempt', first_ftd_attempt_delay_hours_30d, has_ftd_success_30d FROM base
      UNION ALL
      SELECT 'First FTD success', first_ftd_success_delay_hours_30d, has_ftd_success_30d FROM base
    ),
    bucketed AS (
      SELECT
        process,
        CASE
          WHEN delay_hours < 1 THEN '< 1h'
          WHEN delay_hours < 24 THEN '1-24h'
          WHEN delay_hours < 72 THEN '1-3d'
          WHEN delay_hours < 168 THEN '3-7d'
          ELSE '7d+'
        END AS delay_bucket,
        CASE
          WHEN delay_hours < 1 THEN 1
          WHEN delay_hours < 24 THEN 2
          WHEN delay_hours < 72 THEN 3
          WHEN delay_hours < 168 THEN 4
          ELSE 5
        END AS delay_bucket_order,
        converted
      FROM events
      WHERE delay_hours IS NOT NULL
    )
    SELECT
      process,
      delay_bucket,
      delay_bucket_order,
      COUNT(*) AS users,
      AVG(CAST(converted AS DOUBLE)) AS ftd_conversion_rate
    FROM bucketed
    GROUP BY process, delay_bucket, delay_bucket_order
    ORDER BY process, delay_bucket_order
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_sequences(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    SELECT
      ekyc_status_sequence,
      COUNT(*) AS users,
      AVG(CASE WHEN COALESCE(has_ekyc_failure_30d, 0) = 1 THEN CAST(COALESCE(ekyc_recovered_after_failure, 0) AS DOUBLE) END) AS recovery_after_failure_rate,
      AVG(CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE)) AS ftd_conversion_rate
    FROM {table_name}
    {_where_clause(matured_only)}
      {"AND" if matured_only else "WHERE"} ekyc_status_sequence IS NOT NULL
    GROUP BY ekyc_status_sequence
    ORDER BY users DESC, ekyc_status_sequence
    LIMIT 20
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_fail_reasons(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    SELECT
      first_ekyc_fail_reason,
      COUNT(*) AS users,
      AVG(CAST(COALESCE(ekyc_recovered_after_failure, 0) AS DOUBLE)) AS recovery_rate,
      AVG(CAST(COALESCE(has_ftd_attempt_30d, 0) AS DOUBLE)) AS ftd_attempt_rate,
      AVG(CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE)) AS ftd_conversion_rate
    FROM {table_name}
    {_where_clause(matured_only)}
      {"AND" if matured_only else "WHERE"} first_ekyc_fail_reason IS NOT NULL
      AND first_ekyc_fail_reason <> 'Missing'
    GROUP BY first_ekyc_fail_reason
    ORDER BY users DESC, first_ekyc_fail_reason
    LIMIT 25
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_coupon_summary(http_path: str, table_name: str, matured_only: bool) -> pd.DataFrame:
    query = f"""
    SELECT
      CASE WHEN COALESCE(has_coupon_before_first_ftd, 0) = 1 THEN 'Coupon before first FTD attempt' ELSE 'No coupon before first FTD attempt' END AS coupon_group,
      COUNT(*) AS users,
      AVG(CAST(COALESCE(has_ekyc_verified_30d, 0) AS DOUBLE)) AS ekyc_verified_rate,
      AVG(CAST(COALESCE(has_ftd_attempt_30d, 0) AS DOUBLE)) AS ftd_attempt_rate,
      AVG(CAST(COALESCE(has_ftd_success_30d, 0) AS DOUBLE)) AS ftd_conversion_rate,
      AVG(CAST(COALESCE(num_coupons_before_first_ftd, 0) AS DOUBLE)) AS avg_coupons,
      AVG(CAST(COALESCE(coupon_amount_before_first_ftd, 0) AS DOUBLE)) AS avg_coupon_amount
    FROM {table_name}
    {_where_clause(matured_only)}
    GROUP BY CASE WHEN COALESCE(has_coupon_before_first_ftd, 0) = 1 THEN 'Coupon before first FTD attempt' ELSE 'No coupon before first FTD attempt' END
    ORDER BY users DESC
    """
    return data_select(query=query, http_path=http_path)


def get_last_updated(context: dict | None = None):
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        return "-"
    table_name = _resolve_table_name(context, _analysis_name())
    freshness = load_report_freshness(http_path, table_name)
    return freshness.get("report_updated_at") or "-"


def _render_context(verification_history_source: str, coupon_source_available: bool) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Explicit status semantics")
        st.caption("Failure, under review, expired, black list, and verified are kept as separate operational outcomes.")
    with c2:
        st.markdown("##### Observed journey and recovery")
        if "log" in verification_history_source.lower():
            st.caption("Status history is available; the report tracks observed transitions and recovery.")
        else:
            st.caption("No status log was found. Sequences represent verification attempt outcomes, not every internal status transition.")
    with c3:
        st.markdown("##### Direct coupon evidence")
        if coupon_source_available:
            st.caption("Coupon exposure is measured before the first FTD attempt, not through a wallet-event proxy.")
        else:
            st.caption("The direct coupon source was unavailable when this dataset was materialized.")
    with st.expander("Journey definitions used in this report"):
        st.markdown(
            "- **Explicit eKYC failure:** `verified_status = 4` or `fail_reason > 0`; other non-success statuses remain separate.\n"
            "- **eKYC postponement signals:** under review and expired statuses, plus elapsed-time buckets.\n"
            "- **FTD postponement/failure:** pending or handling statuses are separate from failure or timeout.\n"
            "- **FTD journey boundary:** deposit attempts after the first successful deposit are excluded."
        )


def _render_funnel_chart(stage_df: pd.DataFrame) -> None:
    if stage_df.empty:
        return
    chart = (
        alt.Chart(stage_df)
        .mark_bar(size=28, cornerRadiusEnd=4)
        .encode(
            x=alt.X("users:Q", title="Users"),
            y=alt.Y("stage:N", sort=alt.EncodingSortField(field="stage_order", order="ascending"), title=None),
            color=alt.Color("stage:N", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("pct_registered:Q", title="Share of registrations", format=".1%"),
            ],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_monthly_chart(monthly_df: pd.DataFrame) -> None:
    if monthly_df.empty:
        return
    chart_df = monthly_df.melt(
        id_vars=["registration_month"],
        value_vars=["ekyc_start_rate", "ekyc_verified_rate", "explicit_failure_rate", "ftd_conversion_rate"],
        var_name="metric",
        value_name="rate",
    )
    labels = {
        "ekyc_start_rate": "eKYC started",
        "ekyc_verified_rate": "eKYC verified",
        "explicit_failure_rate": "Explicit eKYC failure",
        "ftd_conversion_rate": "FTD converted",
    }
    chart_df["metric"] = chart_df["metric"].map(labels).fillna(chart_df["metric"])
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("registration_month:T", title="Registration month", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("rate:Q", title="Rate"),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                alt.Tooltip("registration_month:T", title="Month", format="%Y-%m"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("rate:Q", title="Rate", format=".1%"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_delay_chart(delay_df: pd.DataFrame) -> None:
    if delay_df.empty:
        return
    chart = (
        alt.Chart(delay_df)
        .mark_bar(size=18)
        .encode(
            x=alt.X(
                "delay_bucket:N",
                sort=["< 1h", "1-24h", "1-3d", "3-7d", "7d+"],
                title="Delay bucket",
                scale=alt.Scale(paddingInner=0.55, paddingOuter=0.35),
            ),
            y=alt.Y("users:Q", title="Users"),
            color=alt.Color("ftd_conversion_rate:Q", title="FTD conversion", scale=alt.Scale(scheme="tealblues")),
            tooltip=[
                alt.Tooltip("process:N", title="Process"),
                alt.Tooltip("delay_bucket:N", title="Delay"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("ftd_conversion_rate:Q", title="FTD conversion", format=".1%"),
            ],
        )
        .properties(width=330, height=170)
        .facet(facet=alt.Facet("process:N", title=None), columns=2)
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, use_container_width=False)


def _flow_metric(flow_paths_df: pd.DataFrame, flow_key: str, column: str):
    if flow_paths_df.empty or "flow_key" not in flow_paths_df.columns:
        return None
    matched = flow_paths_df[flow_paths_df["flow_key"] == flow_key]
    if matched.empty:
        return None
    return matched.iloc[0].get(column)


def _flow_box(title: str, note: str = "", style: str = "") -> str:
    css_class = "ekyc-flow-box"
    if style:
        css_class = f"{css_class} {style}"
    note_html = f'<span class="ekyc-flow-note">{html.escape(note)}</span>' if note else ""
    return (
        f'<div class="{css_class}">'
        f'<span class="ekyc-flow-title">{html.escape(title)}</span>'
        f"{note_html}"
        "</div>"
    )


def _flow_row(steps: list[tuple[str, str, str]]) -> str:
    cells: list[str] = []
    for index, (title, note, style) in enumerate(steps):
        if index > 0:
            cells.append('<div class="ekyc-flow-arrow">→</div>')
        cells.append(_flow_box(title, note, style))
    while len(cells) < 9:
        cells.append("<div></div>")
    return f'<div class="ekyc-flow-row">{"".join(cells[:9])}</div>'


def _share_note(flow_paths_df: pd.DataFrame, flow_key: str) -> str:
    return (
        f"Share of cohort: {_fmt_pct(_flow_metric(flow_paths_df, flow_key, 'population_rate'))} "
        f"· Users: {_fmt_num(_flow_metric(flow_paths_df, flow_key, 'users'))}"
    )


def _render_journey_flow_map(row: pd.Series, flow_paths_df: pd.DataFrame) -> None:
    if flow_paths_df.empty:
        return

    recovery_time = _fmt_hours(row.get("avg_ekyc_recovery_hours"))
    flow_specs = [
        (
            "healthy_converted",
            [
                ("Registration", "", ""),
                ("eKYC approved", "", "ekyc-flow-approved"),
                ("FTD / deposit", "", ""),
                ("Healthy converted", _share_note(flow_paths_df, "healthy_converted"), "ekyc-flow-approved"),
            ],
        ),
        (
            "recovered_converted",
            [
                ("Registration", "", ""),
                ("eKYC failure", "", "ekyc-flow-failure"),
                ("Later approved", f"Avg failure → approval: {recovery_time}", "ekyc-flow-approved"),
                ("FTD / deposit", "", ""),
                ("Recovered and converted", _share_note(flow_paths_df, "recovered_converted"), "ekyc-flow-approved"),
            ],
        ),
        (
            "critical_converted",
            [
                ("Registration", "", ""),
                ("eKYC failure", "", "ekyc-flow-failure"),
                ("No approval observed", "", "ekyc-flow-failure"),
                ("Still deposits", "", ""),
                ("Critical case", _share_note(flow_paths_df, "critical_converted"), "ekyc-flow-critical"),
            ],
        ),
        (
            "converted_without_observed_approval",
            [
                ("Registration", "", ""),
                ("No approval observed", "", "ekyc-flow-failure"),
                ("FTD / deposit", "", ""),
                ("Audit case", _share_note(flow_paths_df, "converted_without_observed_approval"), "ekyc-flow-critical"),
            ],
        ),
        (
            "recovered_no_ftd",
            [
                ("Registration", "", ""),
                ("eKYC failure", "", "ekyc-flow-failure"),
                ("Later approved", f"Avg failure → approval: {recovery_time}", "ekyc-flow-approved"),
                ("No deposit", "", ""),
                ("Recovered, no FTD", _share_note(flow_paths_df, "recovered_no_ftd"), "ekyc-flow-friction"),
            ],
        ),
        (
            "failure_no_ftd",
            [
                ("Registration", "", ""),
                ("eKYC failure", "", "ekyc-flow-failure"),
                ("No approval observed", "", "ekyc-flow-failure"),
                ("No deposit", "", ""),
                ("Real friction", _share_note(flow_paths_df, "failure_no_ftd"), "ekyc-flow-friction"),
            ],
        ),
        (
            "verified_no_ftd",
            [
                ("Registration", "", ""),
                ("eKYC approved", "", "ekyc-flow-approved"),
                ("No deposit", "", ""),
                ("Post-approval drop-off", _share_note(flow_paths_df, "verified_no_ftd"), "ekyc-flow-friction"),
            ],
        ),
        (
            "started_no_ftd",
            [
                ("Registration", "", ""),
                ("eKYC started", "", ""),
                ("No approval / FTD", "", "ekyc-flow-failure"),
                ("Started only", _share_note(flow_paths_df, "started_no_ftd"), "ekyc-flow-friction"),
            ],
        ),
        (
            "no_ekyc_no_ftd",
            [
                ("Registration", "", ""),
                ("No eKYC observed", "", "ekyc-flow-failure"),
                ("No deposit", "", ""),
                ("Pre-eKYC drop-off", _share_note(flow_paths_df, "no_ekyc_no_ftd"), "ekyc-flow-friction"),
            ],
        ),
    ]
    rows_html = "".join(
        _flow_row(steps)
        for flow_key, steps in flow_specs
        if _flow_metric(flow_paths_df, flow_key, "users") is not None
    )

    st.markdown(
        f"""
        <style>
          .ekyc-flow-map {{
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
            margin: 0.6rem 0 1.4rem;
            overflow-x: auto;
          }}
          .ekyc-flow-row {{
            display: grid;
            grid-template-columns: minmax(130px, 0.9fr) 24px minmax(150px, 1fr) 24px minmax(170px, 1.1fr) 24px minmax(150px, 1fr) 24px minmax(190px, 1.35fr);
            align-items: center;
            min-width: 980px;
          }}
          .ekyc-flow-box {{
            border: 1px solid rgba(111, 158, 219, 0.34);
            border-radius: 12px;
            padding: 0.72rem 0.82rem;
            min-height: 54px;
            background: rgba(13, 28, 52, 0.72);
            color: #edf4ff;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
          }}
          .ekyc-flow-title {{
            display: block;
            font-weight: 700;
            line-height: 1.15;
          }}
          .ekyc-flow-note {{
            display: block;
            margin-top: 0.26rem;
            color: #a9b7cc;
            font-size: 0.82rem;
            line-height: 1.25;
          }}
          .ekyc-flow-arrow {{
            color: #7fa5d7;
            font-weight: 800;
            text-align: center;
            font-size: 1rem;
          }}
          .ekyc-flow-approved {{
            border-color: rgba(65, 196, 148, 0.48);
            background: linear-gradient(135deg, rgba(16, 75, 65, 0.7), rgba(13, 40, 53, 0.82));
          }}
          .ekyc-flow-failure {{
            border-color: rgba(255, 117, 91, 0.58);
            background: linear-gradient(135deg, rgba(88, 34, 36, 0.72), rgba(43, 28, 44, 0.82));
          }}
          .ekyc-flow-friction {{
            border-color: rgba(255, 190, 92, 0.58);
            background: linear-gradient(135deg, rgba(91, 65, 25, 0.72), rgba(45, 38, 33, 0.86));
          }}
          .ekyc-flow-critical {{
            border-color: rgba(239, 91, 113, 0.7);
            background: linear-gradient(135deg, rgba(99, 34, 50, 0.74), rgba(47, 30, 48, 0.88));
          }}
        </style>
        <div class="ekyc-flow-map">
          {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_rate_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    show = df.copy()
    for column in columns:
        if column in show.columns:
            show[column] = show[column].apply(_fmt_pct)
    if "users" in show.columns:
        show["users"] = show["users"].apply(_fmt_num)
    return show


def render(context: dict | None = None) -> None:
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        st.error(
            "Missing SQL warehouse HTTP path. Set `DATABRICKS_SQL_HTTP_PATH` "
            "(or `DATABRICKS_HTTP_PATH`) in app environment variables."
        )
        return

    default_analysis = _analysis_name()
    analysis_options = _analysis_options()
    if default_analysis not in analysis_options:
        analysis_options = [default_analysis] + analysis_options
    environment = str(context.get("environment", "other")).strip().lower()
    analysis_key = f"funnel_ekyc_journey_analysis_select_{environment}"
    matured_key = f"funnel_ekyc_journey_matured_only_{environment}"
    if analysis_key not in st.session_state or st.session_state[analysis_key] not in analysis_options:
        st.session_state[analysis_key] = default_analysis if default_analysis in analysis_options else analysis_options[0]

    c1, c2 = st.columns([3, 1])
    with c1:
        analysis_name = st.selectbox(
            "Analysis",
            options=analysis_options,
            key=analysis_key,
            help="Select which conversion funnel analysis to display.",
        )
    with c2:
        matured_only = st.checkbox(
            "Matured cohorts only",
            value=True,
            key=matured_key,
            help="Restricts outcome rates to customers with the complete journey observation window.",
        )

    table_name = _resolve_table_name(context, analysis_name)
    try:
        summary_df = _load_summary(http_path, table_name, matured_only)
        stage_df = _load_funnel_stages(http_path, table_name, matured_only)
        monthly_df = _load_monthly_trend(http_path, table_name, matured_only)
        status_df = _load_status_signals(http_path, table_name, matured_only)
        outcomes_df = _load_journey_outcomes(http_path, table_name, matured_only)
        flow_paths_df = _load_flow_paths(http_path, table_name, matured_only)
        delay_df = _load_delay_buckets(http_path, table_name, matured_only)
        sequences_df = _load_sequences(http_path, table_name, matured_only)
        fail_reasons_df = _load_fail_reasons(http_path, table_name, matured_only)
        coupon_df = _load_coupon_summary(http_path, table_name, matured_only)
    except Exception as exc:
        st.error(
            "Failed to load the eKYC journey dataset. "
            "Run `models/conversion_funnel_deposit/07_ekyc_journey_report` before deploying the app. "
            f"Details: {exc}"
        )
        return

    if summary_df.empty:
        st.warning("No eKYC journey data found.")
        return

    try:
        freshness = load_report_freshness(http_path, table_name)
    except Exception:
        freshness = {}

    row = summary_df.iloc[0]
    journey_window_days = int(row.get("journey_window_days") or 30)
    verification_history_source = str(row.get("verification_history_source") or "user_verification attempt outcomes")
    coupon_source_available = bool(row.get("coupon_source_available") or False)
    st.subheader(f"eKYC-to-FTD Customer Journey ({analysis_name})")
    st.caption(
        f"Explicit operational journey during the first {journey_window_days} days after registration. "
        "The app reads only the materialized ML report table."
    )
    render_report_freshness(freshness)
    _render_context(verification_history_source, coupon_source_available)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", _fmt_num(row.get("users")))
    k2.metric("eKYC started", _fmt_pct(row.get("ekyc_start_rate")))
    k3.metric("eKYC verified", _fmt_pct(row.get("ekyc_verified_rate")))
    k4.metric("Explicit eKYC failure", _fmt_pct(row.get("explicit_failure_rate")))

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Recovered after failure", _fmt_pct(row.get("recovery_after_failure_rate")))
    k6.metric("Under review → verified", _fmt_pct(row.get("review_to_verified_rate")))
    k7.metric("FTD attempted", _fmt_pct(row.get("ftd_attempt_rate")))
    k8.metric("FTD converted", _fmt_pct(row.get("ftd_conversion_rate")))

    k9, k10, k11, k12 = st.columns(4)
    k9.metric("eKYC under review", _fmt_pct(row.get("under_review_rate")))
    k10.metric("eKYC before first FTD", _fmt_pct(row.get("ekyc_before_ftd_attempt_rate")))
    k11.metric("FTD conversion after verified", _fmt_pct(row.get("conversion_after_verified_rate")))
    k12.metric(
        "Coupon before first FTD",
        _fmt_pct(row.get("coupon_before_ftd_rate")) if coupon_source_available else "Unavailable",
    )

    st.markdown("##### Main journey stage reach")
    st.caption("Stages are measured independently and are not forced to be nested.")
    _render_funnel_chart(stage_df)

    st.markdown("##### eKYC-to-FTD journey patterns")
    st.caption("Rows are mutually exclusive. Their shares are relative to the selected cohort and sum to 100%.")
    _render_journey_flow_map(row, flow_paths_df)

    st.markdown("##### Journey trend by registration month")
    _render_monthly_chart(monthly_df)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Operational statuses and recoveries")
        status_show = _format_rate_columns(
            status_df,
            ["population_rate", "ftd_attempt_rate", "ftd_conversion_rate"],
        )
        st.dataframe(status_show, use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### Journey outcomes within window")
        outcome_show = _format_rate_columns(outcomes_df, ["population_rate", "coupon_rate"])
        st.dataframe(outcome_show, use_container_width=True, hide_index=True)

    st.markdown("##### Journey timing")
    st.caption(
        "Delays are measured from registration. Color indicates the eventual FTD conversion rate for customers in each timing bucket."
    )
    _render_delay_chart(delay_df)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Most common observed eKYC sequences")
        sequence_show = _format_rate_columns(
            sequences_df,
            ["recovery_after_failure_rate", "ftd_conversion_rate"],
        )
        st.dataframe(sequence_show, use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### First explicit eKYC failure reason")
        fail_show = _format_rate_columns(
            fail_reasons_df,
            ["recovery_rate", "ftd_attempt_rate", "ftd_conversion_rate"],
        )
        st.dataframe(fail_show, use_container_width=True, hide_index=True)

    st.markdown("##### Coupon before first FTD attempt")
    if coupon_source_available:
        coupon_show = _format_rate_columns(
            coupon_df,
            ["ekyc_verified_rate", "ftd_attempt_rate", "ftd_conversion_rate"],
        )
        if "avg_coupons" in coupon_show.columns:
            coupon_show["avg_coupons"] = coupon_show["avg_coupons"].apply(
                lambda value: "-" if value is None or pd.isna(value) else f"{float(value):.2f}"
            )
        if "avg_coupon_amount" in coupon_show.columns:
            coupon_show["avg_coupon_amount"] = coupon_show["avg_coupon_amount"].apply(
                lambda value: "-" if value is None or pd.isna(value) else f"{float(value):,.2f}"
            )
        st.dataframe(coupon_show, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Direct coupon data was unavailable during materialization. "
            "Grant the job identity access to the coupon source and rerun the journey notebook."
        )

    st.markdown("##### Journey timing interpretation")
    t1, t2, t3 = st.columns(3)
    t1.metric("Avg failure → recovery", _fmt_hours(row.get("avg_ekyc_recovery_hours")))
    t2.metric("Avg under review → verified", _fmt_hours(row.get("avg_review_to_verified_hours")))
    t3.metric("Avg verified → first FTD attempt", _fmt_hours(row.get("avg_verified_to_ftd_attempt_hours")))
