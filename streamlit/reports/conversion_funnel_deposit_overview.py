from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select, resolve_sql_http_path

REPORT_META = {
    "title": "Deposit Conversion Funnel SHAP",
    "schema": "conversion_funnel_deposit",
    # "environments": ("training", "production"),
    "environments": ("training"),
    "description": "Monthly register-to-deposit funnel with model evaluation and SHAP explainability.",
}

FUNNEL_ANALYSIS = os.getenv("FUNNEL_ANALYSIS", "analysis_2")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")

ANALYSIS_NAMES = {
    "analysis_1": "register_to_deposit_30d",
    "analysis_2": "register_to_deposit_30d_truewin",
    "analysis_3": "register_to_deposit_30d_igaming",
}


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


def _resolve_table_names(context: dict, analysis_name: str) -> tuple[str, str, str]:
    environment = str(context.get("environment", "production")).strip().lower()
    env_key = _environment_key(environment)

    default_catalog = "ml_train" if env_key == "TRAINING" else "ml_prod"
    default_funnel = f"{default_catalog}.conversion_funnel_deposit.rpt_funnel_{analysis_name}_current"
    default_shap = f"{default_catalog}.conversion_funnel_deposit.rpt_shap_{analysis_name}_current"
    default_eval = f"{default_catalog}.conversion_funnel_deposit.rpt_eval_{analysis_name}_current"

    funnel_override = str(os.getenv(f"FUNNEL_TABLE_{env_key}", "")).strip() or str(os.getenv("FUNNEL_TABLE", "")).strip()
    shap_override = str(os.getenv(f"FUNNEL_SHAP_TABLE_{env_key}", "")).strip() or str(os.getenv("FUNNEL_SHAP_TABLE", "")).strip()
    eval_override = str(os.getenv(f"FUNNEL_EVAL_TABLE_{env_key}", "")).strip() or str(os.getenv("FUNNEL_EVAL_TABLE", "")).strip()

    funnel_table = _safe_table_name(funnel_override) if funnel_override else _safe_table_name(default_funnel)
    shap_table = _safe_table_name(shap_override) if shap_override else _safe_table_name(default_shap)
    eval_table = _safe_table_name(eval_override) if eval_override else _safe_table_name(default_eval)

    return funnel_table, shap_table, eval_table


@st.cache_data(ttl=180)
def _load_funnel(http_path: str, funnel_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      CAST(registration_month AS DATE) AS registration_month,
      CAST(registrations AS BIGINT) AS registrations,
      CAST(attempted_30d AS BIGINT) AS attempted_30d,
      CAST(success_30d AS BIGINT) AS success_30d,
      CAST(failed_after_attempt_30d AS BIGINT) AS failed_after_attempt_30d,
      CAST(never_attempted_30d AS BIGINT) AS never_attempted_30d,
      CAST(attempt_rate_30d AS DOUBLE) AS attempt_rate_30d,
      CAST(success_rate_30d AS DOUBLE) AS success_rate_30d,
      CAST(success_given_attempt_rate_30d AS DOUBLE) AS success_given_attempt_rate_30d,
      CAST(failed_given_attempt_rate_30d AS DOUBLE) AS failed_given_attempt_rate_30d,
      CAST(matured_registrations AS BIGINT) AS matured_registrations,
      CAST(avg_predicted_positive_rate AS DOUBLE) AS avg_predicted_positive_rate,
      CAST(predicted_positive_users AS BIGINT) AS predicted_positive_users
    FROM {funnel_table}
    ORDER BY registration_month DESC
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_shap(http_path: str, shap_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      feature_name,
      CAST(feature_rank AS INT) AS feature_rank,
      CAST(mean_abs_shap AS DOUBLE) AS mean_abs_shap,
      CAST(mean_shap AS DOUBLE) AS mean_shap,
      CAST(sample_size AS BIGINT) AS sample_size
    FROM {shap_table}
    ORDER BY mean_abs_shap DESC, feature_rank ASC
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_eval(http_path: str, eval_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      split_name,
      split_strategy,
      holdout_months,
      CAST(n_rows AS BIGINT) AS n_rows,
      CAST(positive_rate AS DOUBLE) AS positive_rate,
      CAST(accuracy AS DOUBLE) AS accuracy,
      CAST(precision AS DOUBLE) AS precision,
      CAST(recall AS DOUBLE) AS recall,
      CAST(f1 AS DOUBLE) AS f1,
      CAST(roc_auc AS DOUBLE) AS roc_auc,
      CAST(pr_auc AS DOUBLE) AS pr_auc
    FROM {eval_table}
    ORDER BY CASE split_name WHEN 'test' THEN 1 ELSE 2 END
    """
    return data_select(query=query, http_path=http_path)


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


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
    analysis_widget_key = (
        f"funnel_overview_analysis_select_{str(context.get('environment', 'other')).strip().lower()}"
    )
    if analysis_widget_key not in st.session_state or st.session_state[analysis_widget_key] not in analysis_options:
        st.session_state[analysis_widget_key] = default_analysis if default_analysis in analysis_options else analysis_options[0]

    analysis_name = st.selectbox(
        "Analysis",
        options=analysis_options,
        key=analysis_widget_key,
        help="Select which conversion funnel analysis to display.",
    )

    try:
        funnel_table, shap_table, eval_table = _resolve_table_names(context, analysis_name)
        funnel_df = _load_funnel(http_path, funnel_table)
        shap_df = _load_shap(http_path, shap_table)
        eval_df = _load_eval(http_path, eval_table)
    except Exception as exc:
        st.error(f"Failed to load conversion funnel datasets: {exc}")
        return

    if funnel_df.empty:
        st.warning("No funnel data found.")
        return

    funnel_df["registration_month"] = pd.to_datetime(funnel_df["registration_month"], errors="coerce")
    funnel_df = funnel_df.sort_values("registration_month", ascending=False)
    latest = funnel_df.iloc[0]

    st.subheader(f"Register -> Deposit Funnel ({analysis_name})")
    st.caption("Label window: 30 days from registration. Cohort view is monthly by registration timestamp.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registrations (latest month)", _fmt_num(latest.get("registrations")))
    k2.metric("Attempt rate", _fmt_pct(latest.get("attempt_rate_30d")))
    k3.metric("Success rate", _fmt_pct(latest.get("success_rate_30d")))
    k4.metric("Success | Attempt", _fmt_pct(latest.get("success_given_attempt_rate_30d")))

    trend_df = funnel_df.sort_values("registration_month", ascending=True)[[
        "registration_month",
        "attempt_rate_30d",
        "success_rate_30d",
        "failed_given_attempt_rate_30d",
    ]].copy()
    trend_long = trend_df.melt(
        id_vars=["registration_month"],
        value_vars=["attempt_rate_30d", "success_rate_30d", "failed_given_attempt_rate_30d"],
        var_name="metric",
        value_name="rate",
    )
    metric_label_map = {
        "attempt_rate_30d": "Attempt rate (30d)",
        "success_rate_30d": "Success rate (30d)",
        "failed_given_attempt_rate_30d": "Failed | Attempt rate (30d)",
    }
    trend_long["metric_label"] = trend_long["metric"].map(metric_label_map).fillna(trend_long["metric"])
    st.markdown("##### Funnel trend by cohort month")
    trend_chart = (
        alt.Chart(trend_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("registration_month:T", title="Cohort month", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("rate:Q", title="Rate"),
            color=alt.Color(
                "metric_label:N",
                title="Metric",
                scale=alt.Scale(
                    domain=["Attempt rate (30d)", "Success rate (30d)", "Failed | Attempt rate (30d)"],
                    range=["#4C78A8", "#2EBD85", "#E76F51"],
                ),
            ),
            tooltip=[
                alt.Tooltip("registration_month:T", title="Cohort month", format="%Y-%m"),
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("rate:Q", title="Rate", format=".2%"),
            ],
        )
    )
    st.altair_chart(trend_chart, use_container_width=True)

    st.markdown("##### Cohort table")
    display_df = funnel_df.sort_values("registration_month", ascending=False).copy()
    display_df["registration_month"] = display_df["registration_month"].dt.date
    for col in [
        "attempt_rate_30d",
        "success_rate_30d",
        "success_given_attempt_rate_30d",
        "failed_given_attempt_rate_30d",
        "avg_predicted_positive_rate",
    ]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_fmt_pct)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if not eval_df.empty:
        st.markdown("##### Model evaluation")
        eval_show = eval_df.copy()
        for col in ["positive_rate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            eval_show[col] = eval_show[col].apply(_fmt_pct)
        st.dataframe(eval_show, use_container_width=True, hide_index=True)

    if not shap_df.empty:
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        total_abs_shap = float(shap_df["mean_abs_shap"].fillna(0.0).sum())
        if total_abs_shap > 0:
            shap_df["importance_pct"] = shap_df["mean_abs_shap"].fillna(0.0) / total_abs_shap
            shap_df["cum_importance_pct"] = shap_df["importance_pct"].cumsum()
        else:
            shap_df["importance_pct"] = 0.0
            shap_df["cum_importance_pct"] = 0.0

        st.markdown("##### SHAP top features")
        max_top_n = min(30, len(shap_df))
        default_top_n = min(12, max_top_n)
        top_n = st.slider("Top features", min_value=1, max_value=max_top_n, value=default_top_n)
        top_shap = shap_df.head(top_n).copy()
        chart_df = top_shap[["feature_name", "mean_abs_shap"]].sort_values("mean_abs_shap", ascending=False)
        shap_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
                y=alt.Y(
                    "feature_name:N",
                    sort=alt.EncodingSortField(field="mean_abs_shap", op="sum", order="descending"),
                    title="Feature",
                ),
            )
        )
        st.altair_chart(shap_chart, use_container_width=True)
        display_shap = top_shap.copy()
        display_shap["importance_pct"] = display_shap["importance_pct"].apply(_fmt_pct)
        display_shap["cum_importance_pct"] = display_shap["cum_importance_pct"].apply(_fmt_pct)
        st.dataframe(display_shap, use_container_width=True, hide_index=True)
