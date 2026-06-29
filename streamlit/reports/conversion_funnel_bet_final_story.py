from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import resolve_sql_http_path
from reports import conversion_funnel_bet_demographic as demographic_report
from reports import conversion_funnel_bet_deposit_journey as journey_report
from reports import conversion_funnel_bet_overview as overview_report
from reports._funnel_metadata import load_report_freshness, render_report_freshness

REPORT_META = {
    "title": "Bet Conversion Funnel Executive Story",
    "schema": "conversion_funnel_bet",
    "environments": ("training",),
    "description": "Executive view of activation from first successful deposit to first valid wager.",
}

ANALYSIS_NAMES = overview_report.ANALYSIS_NAMES
DEFAULT_ANALYSIS_ID = os.getenv("BET_FUNNEL_FINAL_ANALYSIS", "analysis_2")

FEATURE_ALIASES = {
    "first_deposit_amount": "First deposit amount",
    "first_deposit_channel_type": "First deposit channel",
    "first_deposit_mode_type": "First deposit mode",
    "deposit_hour": "First deposit hour",
    "deposit_weekday": "First deposit weekday",
    "deposit_day_of_month": "First deposit day of month",
    "deposit_weekend_flag": "Weekend first deposit",
    "registration_to_deposit_hours": "Time from registration to deposit",
    "customer_age_years": "Customer age",
    "customer_nationality": "Customer nationality",
    "registered_by_mobile": "Registration device",
    "registered_by_desktop": "Registration device",
}


def _analysis_name() -> str:
    override = str(os.getenv("BET_FUNNEL_FINAL_ANALYSIS_NAME", "")).strip()
    if override:
        return override
    return ANALYSIS_NAMES.get(DEFAULT_ANALYSIS_ID, "deposit_to_wager_30d_truewin")


def _analysis_options() -> list[str]:
    return list(dict.fromkeys(ANALYSIS_NAMES[key] for key in sorted(ANALYSIS_NAMES)))


def _fmt_pct(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):.1%}"


def _fmt_num(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{int(value):,}"


def _fmt_hours(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{float(value):,.1f} h"


def _feature_label(name: str) -> str:
    return FEATURE_ALIASES.get(str(name), str(name).replace("_", " ").title())


def _top_segment_rows(segments_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame()
    available = segments_df[segments_df["feature_name"].isin(feature_names)].copy()
    if available.empty:
        return available

    available["users"] = pd.to_numeric(available["users"], errors="coerce").fillna(0.0)
    available["conversion_rate"] = pd.to_numeric(available["conversion_rate"], errors="coerce")
    available = available.sort_values(["feature_name", "users"], ascending=[True, False])
    top = available.groupby("feature_name", as_index=False).head(1).copy()
    top["feature"] = top["feature_name"].map(_feature_label)
    top["segment"] = top["segment_value"].astype(str)
    top["conversion"] = top["conversion_rate"].apply(_fmt_pct)
    top["users"] = top["users"].apply(_fmt_num)
    return top[["feature", "segment", "users", "conversion"]]


def get_last_updated(context: dict | None = None):
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        return "-"
    table_name = journey_report._resolve_table_name(context, _analysis_name())
    return load_report_freshness(http_path, table_name).get("report_updated_at") or "-"


def render(context: dict | None = None) -> None:
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        st.error("Missing SQL warehouse HTTP path.")
        return

    options = _analysis_options()
    environment = str(context.get("environment", "other")).lower()
    c1, c2 = st.columns([3, 1])
    with c1:
        analysis_name = st.selectbox(
            "Analysis",
            options=options,
            index=options.index(_analysis_name()) if _analysis_name() in options else 0,
            key=f"funnel_bet_story_analysis_{environment}",
        )
    with c2:
        matured_only = st.checkbox(
            "Matured cohorts only",
            value=True,
            key=f"funnel_bet_story_matured_{environment}",
        )

    try:
        funnel_table, shap_table, eval_table = overview_report._resolve_table_names(context, analysis_name)
        journey_table = journey_report._resolve_table_name(context, analysis_name)
        demographic_summary_table, demographic_segments_table = demographic_report._resolve_table_names(
            context,
            analysis_name,
        )
        funnel_df = overview_report._load_funnel(http_path, funnel_table)
        shap_df = overview_report._load_shap(http_path, shap_table)
        eval_df = overview_report._load_eval(http_path, eval_table)
        summary_df = journey_report._load_summary(http_path, journey_table, matured_only)
        trend_df = journey_report._load_trend(http_path, journey_table, matured_only)
        source_df = journey_report._load_sources(http_path, journey_table, matured_only)
        demographic_summary_df = demographic_report._load_summary(http_path, demographic_summary_table)
        demographic_segments_df = demographic_report._load_segments(http_path, demographic_segments_table)
    except Exception as exc:
        st.error(f"Failed to load bet funnel executive story: {exc}")
        return

    if summary_df.empty:
        st.warning("No deposit-to-wager journey data found.")
        return

    row = summary_df.iloc[0]
    depositors = float(row.get("depositors") or 0.0)
    wagered = float(row.get("wagered_30d") or 0.0)
    not_wagered = max(0.0, depositors - wagered)
    conversion = float(row.get("conversion_rate_30d") or 0.0)
    median_delay = row.get("median_hours_to_wager")

    st.subheader(f"Bet Funnel Executive Story ({analysis_name})")
    st.caption(
        "Observed customer activation from first successful deposit to first valid wager in 30 days. "
        "Model signals are associative and should not be interpreted as causal effects."
    )
    try:
        render_report_freshness(load_report_freshness(http_path, journey_table))
    except Exception:
        pass

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("First depositors", _fmt_num(depositors))
    k2.metric("Reached first wager", _fmt_num(wagered))
    k3.metric("Wager conversion", _fmt_pct(conversion))
    k4.metric("Median activation time", _fmt_hours(median_delay))

    st.markdown("#### 1. Where does activation stop?")
    funnel_stage_df = pd.DataFrame(
        {
            "stage": ["First successful deposit", "First wager within 30d", "No wager within 30d"],
            "users": [depositors, wagered, not_wagered],
        }
    )
    st.altair_chart(
        alt.Chart(funnel_stage_df)
        .mark_bar()
        .encode(
            x=alt.X("users:Q", title="Users"),
            y=alt.Y(
                "stage:N",
                sort=["First successful deposit", "First wager within 30d", "No wager within 30d"],
                title=None,
            ),
            color=alt.Color(
                "stage:N",
                legend=None,
                scale=alt.Scale(range=["#4C78A8", "#2EBD85", "#E76F51"]),
            ),
            tooltip=["stage:N", alt.Tooltip("users:Q", format=",.0f")],
        )
        .properties(height=220),
        use_container_width=True,
    )
    st.info(
        f"{_fmt_num(not_wagered)} depositors did not place a valid wager within 30 days. "
        f"Among converters, the median activation time was {_fmt_hours(median_delay)}."
    )

    st.markdown("#### 2. Is activation improving?")
    if trend_df.empty:
        st.info("No monthly trend data found.")
    else:
        trend_df["deposit_month"] = pd.to_datetime(trend_df["deposit_month"], errors="coerce")
        st.altair_chart(
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("deposit_month:T", title="First-deposit month"),
                y=alt.Y("conversion_rate_30d:Q", title="Wager conversion", axis=alt.Axis(format="%")),
                tooltip=[
                    "deposit_month:T",
                    "depositors:Q",
                    alt.Tooltip("conversion_rate_30d:Q", format=".1%"),
                    alt.Tooltip("median_hours_to_wager:Q", format=",.1f"),
                ],
            )
            .properties(height=300),
            use_container_width=True,
        )

    st.markdown("#### 3. Where are the largest differences?")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Source systems")
        if source_df.empty:
            st.info("No source comparison found.")
        else:
            source_show = source_df.copy()
            source_show["conversion_rate"] = source_show["conversion_rate"].apply(_fmt_pct)
            source_show["median_hours_to_wager"] = source_show["median_hours_to_wager"].apply(_fmt_hours)
            st.dataframe(source_show, use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### Dominant customer segments")
        if demographic_summary_df.empty or demographic_segments_df.empty:
            st.info("No demographic segmentation found.")
        else:
            ranked_features = (
                demographic_summary_df
                .sort_values(["feature_rank", "mean_abs_shap"], ascending=[True, False])
                .head(8)["feature_name"]
                .astype(str)
                .tolist()
            )
            top_segments = _top_segment_rows(demographic_segments_df, ranked_features)
            st.dataframe(top_segments, use_container_width=True, hide_index=True)

    st.markdown("#### 4. What does the model highlight?")
    model_left, model_right = st.columns([1.4, 1])
    with model_left:
        if shap_df.empty:
            st.info("No SHAP data found.")
        else:
            shap_show = shap_df.head(15).copy()
            shap_show["feature_label"] = shap_show["feature_name"].map(_feature_label)
            shap_show = shap_show.sort_values("mean_abs_shap")
            st.altair_chart(
                alt.Chart(shap_show)
                .mark_bar()
                .encode(
                    x=alt.X("mean_abs_shap:Q", title="Mean absolute SHAP"),
                    y=alt.Y("feature_label:N", sort=None, title=None),
                    tooltip=["feature_label:N", alt.Tooltip("mean_abs_shap:Q", format=".5f")],
                )
                .properties(height=420),
                use_container_width=True,
            )
    with model_right:
        if eval_df.empty:
            st.info("No evaluation data found.")
        else:
            test_rows = eval_df[eval_df["split_name"].astype(str).str.lower() == "test"]
            evaluation = test_rows.iloc[0] if not test_rows.empty else eval_df.iloc[0]
            st.metric("Test ROC AUC", _fmt_pct(evaluation.get("roc_auc")))
            st.metric("Test PR AUC", _fmt_pct(evaluation.get("pr_auc")))
            st.metric("Test recall", _fmt_pct(evaluation.get("recall")))
            st.caption(
                f"Split: {evaluation.get('split_strategy', '-')} | "
                f"Rows: {_fmt_num(evaluation.get('n_rows'))}"
            )

    st.caption(
        "Recommended interpretation: use the journey metrics to locate operational friction, "
        "then use SHAP and segment tables to prioritize hypotheses for controlled experiments."
    )
