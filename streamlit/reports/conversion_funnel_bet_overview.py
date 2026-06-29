from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select, resolve_sql_http_path
from reports._funnel_metadata import load_report_freshness, render_report_freshness

REPORT_META = {
    "title": "Bet Conversion Funnel SHAP",
    "schema": "conversion_funnel_bet",
    "environments": ("training",),
    "description": "Monthly deposit-to-first-wager funnel with model evaluation and SHAP explainability.",
}

BET_FUNNEL_ANALYSIS = os.getenv("BET_FUNNEL_ANALYSIS", "analysis_2")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")

ANALYSIS_NAMES = {
    "analysis_1": "deposit_to_wager_30d",
    "analysis_2": "deposit_to_wager_30d_truewin",
    "analysis_3": "deposit_to_wager_30d_igaming",
    "analysis_4": "deposit_to_wager_30d_truewin_no_ekyc",
    "analysis_5": "deposit_to_wager_30d_truewin_no_ekyc_no_remarks_coupon",
}


def _environment_key(environment: str) -> str:
    env = str(environment).strip().lower()
    if env in {"training", "train", "ml_train"}:
        return "TRAINING"
    if env in {"production", "prod", "ml_prod"}:
        return "PRODUCTION"
    return "OTHER"


def _analysis_name() -> str:
    override = str(os.getenv("BET_FUNNEL_ANALYSIS_NAME", "")).strip()
    if override:
        return override
    return ANALYSIS_NAMES.get(BET_FUNNEL_ANALYSIS, "deposit_to_wager_30d_truewin")


def _analysis_options() -> list[str]:
    return list(dict.fromkeys(ANALYSIS_NAMES[key] for key in sorted(ANALYSIS_NAMES)))


def _safe_table_name(name: str) -> str:
    table_name = str(name).strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table identifier: {table_name}")
    return table_name


def _resolve_table_names(context: dict, analysis_name: str) -> tuple[str, str, str]:
    env_key = _environment_key(str(context.get("environment", "production")))
    default_catalog = "ml_train" if env_key == "TRAINING" else "ml_prod"
    schema = "conversion_funnel_bet"

    defaults = (
        f"{default_catalog}.{schema}.rpt_funnel_{analysis_name}_current",
        f"{default_catalog}.{schema}.rpt_shap_{analysis_name}_current",
        f"{default_catalog}.{schema}.rpt_eval_{analysis_name}_current",
    )
    overrides = (
        str(os.getenv(f"BET_FUNNEL_TABLE_{env_key}", "")).strip() or str(os.getenv("BET_FUNNEL_TABLE", "")).strip(),
        str(os.getenv(f"BET_FUNNEL_SHAP_TABLE_{env_key}", "")).strip() or str(os.getenv("BET_FUNNEL_SHAP_TABLE", "")).strip(),
        str(os.getenv(f"BET_FUNNEL_EVAL_TABLE_{env_key}", "")).strip() or str(os.getenv("BET_FUNNEL_EVAL_TABLE", "")).strip(),
    )
    return tuple(_safe_table_name(override or default) for override, default in zip(overrides, defaults))


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_funnel(http_path: str, table_name: str) -> pd.DataFrame:
    query = f"""
    SELECT
      CAST(deposit_month AS DATE) AS deposit_month,
      CAST(depositors AS BIGINT) AS depositors,
      CAST(wagered_30d AS BIGINT) AS wagered_30d,
      CAST(not_wagered_30d AS BIGINT) AS not_wagered_30d,
      CAST(matured_depositors AS BIGINT) AS matured_depositors,
      CAST(matured_wagered_30d AS BIGINT) AS matured_wagered_30d,
      CAST(wager_conversion_rate_30d AS DOUBLE) AS wager_conversion_rate_30d,
      CAST(matured_wager_conversion_rate_30d AS DOUBLE) AS matured_wager_conversion_rate_30d,
      CAST(avg_first_wager_delay_hours_30d AS DOUBLE) AS avg_first_wager_delay_hours_30d,
      CAST(median_first_wager_delay_hours_30d AS DOUBLE) AS median_first_wager_delay_hours_30d,
      CAST(avg_predicted_positive_rate AS DOUBLE) AS avg_predicted_positive_rate,
      CAST(predicted_positive_users AS BIGINT) AS predicted_positive_users
    FROM {table_name}
    ORDER BY deposit_month DESC
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_shap(http_path: str, table_name: str) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT feature_name, CAST(feature_rank AS INT) AS feature_rank,
               CAST(mean_abs_shap AS DOUBLE) AS mean_abs_shap,
               CAST(mean_shap AS DOUBLE) AS mean_shap,
               CAST(sample_size AS BIGINT) AS sample_size
        FROM {table_name}
        ORDER BY mean_abs_shap DESC, feature_rank ASC
        """,
        http_path=http_path,
    )


@st.cache_data(ttl=180, show_spinner="Loading...")
def _load_eval(http_path: str, table_name: str) -> pd.DataFrame:
    return data_select(
        query=f"""
        SELECT split_name, split_strategy, holdout_months,
               CAST(n_rows AS BIGINT) AS n_rows,
               CAST(positive_rate AS DOUBLE) AS positive_rate,
               CAST(accuracy AS DOUBLE) AS accuracy,
               CAST(precision AS DOUBLE) AS precision,
               CAST(recall AS DOUBLE) AS recall,
               CAST(f1 AS DOUBLE) AS f1,
               CAST(roc_auc AS DOUBLE) AS roc_auc,
               CAST(pr_auc AS DOUBLE) AS pr_auc
        FROM {table_name}
        ORDER BY CASE split_name WHEN 'test' THEN 1 ELSE 2 END
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
    funnel_table, _, _ = _resolve_table_names(context, _analysis_name())
    return load_report_freshness(http_path, funnel_table).get("report_updated_at") or "-"


def render(context: dict | None = None) -> None:
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        st.error("Missing SQL warehouse HTTP path.")
        return

    options = _analysis_options()
    default_analysis = _analysis_name()
    key = f"funnel_bet_overview_analysis_{str(context.get('environment', 'other')).lower()}"
    analysis_name = st.selectbox(
        "Analysis",
        options=options,
        index=options.index(default_analysis) if default_analysis in options else 0,
        key=key,
    )

    try:
        funnel_table, shap_table, eval_table = _resolve_table_names(context, analysis_name)
        funnel_df = _load_funnel(http_path, funnel_table)
        shap_df = _load_shap(http_path, shap_table)
        eval_df = _load_eval(http_path, eval_table)
    except Exception as exc:
        st.error(f"Failed to load deposit-to-wager datasets: {exc}")
        return

    if funnel_df.empty:
        st.warning("No funnel data found.")
        return

    funnel_df["deposit_month"] = pd.to_datetime(funnel_df["deposit_month"], errors="coerce")
    funnel_df = funnel_df.sort_values("deposit_month", ascending=False)
    latest = funnel_df.iloc[0]

    st.subheader(f"Deposit -> First Wager Funnel ({analysis_name})")
    st.caption("Conversion means at least one valid wager placement within 30 days of the first successful deposit.")
    try:
        render_report_freshness(load_report_freshness(http_path, funnel_table))
    except Exception:
        pass

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("First depositors", _fmt_num(latest.get("depositors")))
    k2.metric("Wagered within 30d", _fmt_pct(latest.get("wager_conversion_rate_30d")))
    k3.metric("Matured conversion", _fmt_pct(latest.get("matured_wager_conversion_rate_30d")))
    k4.metric("Median time to wager", _fmt_hours(latest.get("median_first_wager_delay_hours_30d")))

    trend = funnel_df.sort_values("deposit_month")
    trend_long = trend.melt(
        id_vars=["deposit_month"],
        value_vars=["wager_conversion_rate_30d", "matured_wager_conversion_rate_30d", "avg_predicted_positive_rate"],
        var_name="metric",
        value_name="rate",
    )
    labels = {
        "wager_conversion_rate_30d": "Observed conversion",
        "matured_wager_conversion_rate_30d": "Matured conversion",
        "avg_predicted_positive_rate": "Average predicted propensity",
    }
    trend_long["metric"] = trend_long["metric"].map(labels)
    st.markdown("##### Conversion trend by first-deposit month")
    st.altair_chart(
        alt.Chart(trend_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("deposit_month:T", title="First-deposit month"),
            y=alt.Y("rate:Q", title="Rate", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["deposit_month:T", "metric:N", alt.Tooltip("rate:Q", format=".1%")],
        )
        .properties(height=320),
        use_container_width=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("##### Global SHAP importance")
        if shap_df.empty:
            st.info("No SHAP data found.")
        else:
            shap_show = shap_df.head(20).sort_values("mean_abs_shap")
            st.altair_chart(
                alt.Chart(shap_show)
                .mark_bar()
                .encode(
                    x=alt.X("mean_abs_shap:Q", title="Mean absolute SHAP"),
                    y=alt.Y("feature_name:N", sort=None, title=None),
                    tooltip=["feature_name:N", alt.Tooltip("mean_abs_shap:Q", format=".5f")],
                )
                .properties(height=520),
                use_container_width=True,
            )
    with right:
        st.markdown("##### Model evaluation")
        if eval_df.empty:
            st.info("No evaluation data found.")
        else:
            show = eval_df.copy()
            for column in ["positive_rate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                if column in show.columns:
                    show[column] = show[column].apply(_fmt_pct)
            st.dataframe(show, use_container_width=True, hide_index=True)
