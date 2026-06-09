from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import resolve_sql_http_path
from reports import conversion_funnel_deposit_demographic as demographic_report
from reports import conversion_funnel_deposit_ekyc_journey as ekyc_report
from reports import conversion_funnel_deposit_overview as overview_report
from reports._funnel_metadata import load_report_freshness, render_report_freshness

REPORT_META = {
    "title": "Deposit Conversion Funnel Executive Story",
    "schema": "conversion_funnel_deposit",
    "environments": ("training",),
    "description": "Business-oriented conclusion report focused on where conversion friction happens before and after eKYC.",
}

ANALYSIS_NAMES = overview_report.ANALYSIS_NAMES
DEFAULT_ANALYSIS_ID = os.getenv("FUNNEL_FINAL_ANALYSIS", "analysis_5")
DEFAULT_TOP_SHAP = 12

DEMOGRAPHIC_STORY_FEATURES = [
    "registration_hour",
    "registration_weekday",
    "registration_day_of_month",
    "registration_weekend_flag",
    "registration_device",
    "customer_age_years",
    "customer_nationality",
]

FEATURE_BUSINESS_ALIASES = {
    "registration_hour": "Registration time of day",
    "registration_weekday": "Registration weekday",
    "registration_day_of_month": "Registration day of month",
    "registration_weekend_flag": "Weekend registration",
    "registration_device": "Registration device",
    "registered_by_mobile": "Registration device",
    "registered_by_desktop": "Registration device",
    "num_login_methods": "Login method diversity",
    "num_login_events_24h": "Login activity in first 24h",
    "num_distinct_login_ip_24h": "Distinct login IPs in first 24h",
    "num_distinct_login_channel_24h": "Distinct login channels in first 24h",
    "first_coupon_delay_hours_predeposit_30d": "Time until first coupon before deposit",
    "sum_coupon_amount_predeposit_30d": "Coupon value before first deposit",
    "num_coupon_templates_predeposit_30d": "Coupon template variety",
    "num_coupons_predeposit_30d": "Coupon count before first deposit",
    "num_coupon_event_ids_predeposit_30d": "Coupon campaign/event variety",
    "num_coupon_statuses_predeposit_30d": "Coupon status variety",
    "num_distinct_group_types": "User group variety",
    "customer_age_years": "Customer age",
    "customer_nationality": "Customer nationality",
}

JOURNEY_STAGE_LABELS = {
    "pre_ekyc": "Before eKYC",
    "post_approval": "After eKYC approval / deposit activation",
    "customer_profile": "Customer profile",
    "other": "Other model signals",
}

JOURNEY_STAGE_CAPTIONS = {
    "pre_ekyc": "Signals available around registration and early activity. Useful for the registration → eKYC friction story.",
    "post_approval": "Activation and incentive signals before first deposit. Useful for the eKYC approved → deposit story, but still descriptive.",
    "customer_profile": "Observed customer attributes. Descriptive only; they should not be read as causal levers.",
    "other": "Signals that do not map cleanly to one business moment.",
}

EXECUTIVE_FLOW_KEYS = [
    "no_ekyc_no_ftd",
    "healthy_converted",
    "recovered_converted",
    "recovered_no_ftd",
    "failure_no_ftd",
    "verified_no_ftd",
    "critical_converted",
]


def _analysis_name() -> str:
    override = str(os.getenv("FUNNEL_FINAL_ANALYSIS_NAME", "")).strip()
    if override:
        return override
    return ANALYSIS_NAMES.get(DEFAULT_ANALYSIS_ID, DEFAULT_ANALYSIS_ID)


def _analysis_options() -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for _, name in sorted(ANALYSIS_NAMES.items()):
        if name not in seen:
            options.append(name)
            seen.add(name)
    return options


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def _fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{int(digits)}f}"


def _flow_value(flow_paths_df: pd.DataFrame, flow_key: str, column: str):
    if flow_paths_df.empty or "flow_key" not in flow_paths_df.columns:
        return None
    matched = flow_paths_df[flow_paths_df["flow_key"] == flow_key]
    if matched.empty:
        return None
    return matched.iloc[0].get(column)


def _feature_label(feature_name: str) -> str:
    return demographic_report._display_feature_name(feature_name)


def _business_feature_label(feature_name: str) -> str:
    name = str(feature_name)
    if name in FEATURE_BUSINESS_ALIASES:
        return FEATURE_BUSINESS_ALIASES[name]
    return " ".join(part.capitalize() for part in name.replace("_", " ").split())


def _feature_journey_stage(feature_name: str) -> str:
    name = str(feature_name).lower()
    if any(token in name for token in ("coupon", "bonus", "promo", "promotion")):
        return "post_approval"
    if any(token in name for token in ("age", "nationality", "gender", "country", "region")):
        return "customer_profile"
    if any(
        token in name
        for token in (
            "registration",
            "registered",
            "login",
            "device",
            "platform",
            "channel",
            "mail",
            "phone",
            "ip",
            "referrer",
            "marketing",
            "source",
            "browser",
            "os",
        )
    ):
        return "pre_ekyc"
    return "other"


def _parse_segment_numbers(value: str) -> list[float]:
    out: list[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", str(value)):
        try:
            out.append(float(match))
        except ValueError:
            continue
    return out


def _is_underage_segment(raw_value: str) -> bool:
    value = str(raw_value).strip()
    if value in {"__MISSING__", "__OTHER__", "__OUTSIDE__"}:
        return False

    numbers = _parse_segment_numbers(value)
    if not numbers:
        return False

    if value.startswith("[") and len(numbers) >= 2:
        lower, upper = numbers[0], numbers[1]
        return lower < 22 and upper <= 22

    return len(numbers) == 1 and numbers[0] < 22


def _is_unknown_nationality_segment(raw_value: str) -> bool:
    value = str(raw_value).strip().lower()
    return value in {"unknown", "__missing__", "missing", "none", "nan", ""}


def _apply_business_segment_filters(fdf: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    if fdf.empty:
        return fdf

    out = fdf.copy()
    if feature_name == "customer_age_years" and "segment_value" in out.columns:
        out = out[~out["segment_value"].astype(str).apply(_is_underage_segment)].copy()

    if feature_name == "customer_nationality" and "segment_value" in out.columns:
        out = out[~out["segment_value"].astype(str).apply(_is_unknown_nationality_segment)].copy()

    if out.empty:
        return out

    out["users"] = pd.to_numeric(out["users"], errors="coerce").fillna(0.0)
    out["converted_users"] = pd.to_numeric(out["converted_users"], errors="coerce").fillna(0.0)
    out["non_converted_users"] = (out["users"] - out["converted_users"]).clip(lower=0.0)
    total_converted = float(out["converted_users"].sum())
    total_non_converted = float(out["non_converted_users"].sum())
    out["share_in_converted"] = out["converted_users"] / total_converted if total_converted > 0 else 0.0
    out["share_in_non_converted"] = out["non_converted_users"] / total_non_converted if total_non_converted > 0 else 0.0

    return out


def _filter_executive_flow_paths(flow_paths_df: pd.DataFrame) -> pd.DataFrame:
    if flow_paths_df.empty or "flow_key" not in flow_paths_df.columns:
        return flow_paths_df

    filtered = flow_paths_df[flow_paths_df["flow_key"].isin(EXECUTIVE_FLOW_KEYS)].copy()
    if filtered.empty:
        return filtered
    filtered["_flow_order"] = filtered["flow_key"].map(
        {flow_key: index for index, flow_key in enumerate(EXECUTIVE_FLOW_KEYS)}
    )
    return filtered.sort_values("_flow_order").drop(columns=["_flow_order"])


def _render_business_css() -> None:
    st.markdown(
        """
        <style>
          .business-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 12px 0 8px;
          }
          .business-card {
            border: 1px solid rgba(111, 158, 219, 0.32);
            border-radius: 14px;
            padding: 15px 16px;
            background: linear-gradient(135deg, rgba(16, 28, 48, 0.92), rgba(10, 21, 38, 0.94));
            min-height: 122px;
          }
          .business-card-good {
            border-color: rgba(65, 196, 148, 0.44);
          }
          .business-card-warn {
            border-color: rgba(255, 190, 92, 0.52);
          }
          .business-card-risk {
            border-color: rgba(239, 91, 113, 0.52);
          }
          .business-card-label {
            color: #8fa1bb;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .business-card-value {
            color: #f1f6ff;
            display: block;
            font-size: 24px;
            font-weight: 800;
            line-height: 1.15;
            margin: 7px 0 5px;
          }
          .business-card-copy {
            color: #b7c4d8;
            display: block;
            font-size: 13px;
            line-height: 1.35;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _business_card(title: str, value: str, body: str, css_class: str = "") -> str:
    return (
        f'<div class="business-card {css_class}">'
        f'<span class="business-card-label">{title}</span>'
        f'<span class="business-card-value">{value}</span>'
        f'<span class="business-card-copy">{body}</span>'
        "</div>"
    )


def _available_demographic_features(summary_df: pd.DataFrame, segments_df: pd.DataFrame) -> list[str]:
    if summary_df.empty or segments_df.empty or "feature_name" not in summary_df.columns:
        return []

    available_summary = set(summary_df["feature_name"].astype(str).tolist())
    available_segments = set(segments_df["feature_name"].astype(str).tolist())
    available = available_summary | available_segments

    features: list[str] = []
    for feature_name in DEMOGRAPHIC_STORY_FEATURES:
        if feature_name == "registration_device":
            if {"registered_by_mobile", "registered_by_desktop"} & available:
                features.append(feature_name)
        elif feature_name in available:
            features.append(feature_name)
    return features


def get_last_updated(context: dict | None = None):
    context = context or {}
    http_path = resolve_sql_http_path()
    if not http_path:
        return "-"

    table_name = ekyc_report._resolve_table_name(context, _analysis_name())
    freshness = load_report_freshness(http_path, table_name)
    return freshness.get("report_updated_at") or "-"


def _render_readout(row: pd.Series, flow_paths_df: pd.DataFrame) -> None:
    st.markdown("#### Executive story")
    st.caption(
        "Business conclusion: eKYC approval itself is not the main bottleneck. "
        "The largest opportunities are activation before eKYC starts and deposit conversion after eKYC approval."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", _fmt_num(row.get("users")))
    k2.metric("eKYC verified", _fmt_pct(row.get("ekyc_verified_rate")))
    k3.metric("FTD converted", _fmt_pct(row.get("ftd_conversion_rate")))
    k4.metric("FTD after verified", _fmt_pct(row.get("conversion_after_verified_rate")))

    pre_share = _fmt_pct(_flow_value(flow_paths_df, "no_ekyc_no_ftd", "population_rate"))
    pre_users = _fmt_num(_flow_value(flow_paths_df, "no_ekyc_no_ftd", "users"))
    post_share = _fmt_pct(_flow_value(flow_paths_df, "verified_no_ftd", "population_rate"))
    post_users = _fmt_num(_flow_value(flow_paths_df, "verified_no_ftd", "users"))
    unresolved_share = _fmt_pct(_flow_value(flow_paths_df, "failure_no_ftd", "population_rate"))
    unresolved_users = _fmt_num(_flow_value(flow_paths_df, "failure_no_ftd", "users"))
    recovered_share = _fmt_pct(_flow_value(flow_paths_df, "recovered_converted", "population_rate"))
    recovered_users = _fmt_num(_flow_value(flow_paths_df, "recovered_converted", "users"))

    st.markdown(
        '<div class="business-card-grid">'
        + _business_card(
            "What works",
            _fmt_pct(row.get("conversion_after_verified_rate")),
            "Once users are approved, conversion is strong. This suggests approval is not the main blocker.",
            "business-card-good",
        )
        + _business_card(
            "Before eKYC opportunity",
            f"{pre_share}",
            f"{pre_users} registered users do not reach eKYC/FTD. This is the first major friction point.",
            "business-card-warn",
        )
        + _business_card(
            "After approval opportunity",
            f"{post_share}",
            f"{post_users} approved users still do not deposit. This is the second major friction point.",
            "business-card-warn",
        )
        + _business_card(
            "True eKYC friction",
            f"{unresolved_share}",
            f"{unresolved_users} users fail eKYC and do not recover. This is real, but smaller than the two activation gaps.",
            "business-card-risk",
        )
        + _business_card(
            "Recovery path",
            f"{recovered_share}",
            f"{recovered_users} users fail first, recover, and still convert. Recovery appears commercially valuable.",
            "business-card-good",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _build_observed_segment_rows(
    top_shap_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    *,
    direction: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    if top_shap_df.empty or segments_df.empty:
        return pd.DataFrame(rows)

    for _, shap_row in top_shap_df.iterrows():
        feature_name = str(shap_row.get("feature_name"))
        fdf = demographic_report._prepare_segments_feature_view(segments_df, feature_name)
        fdf = _apply_business_segment_filters(fdf, feature_name)
        if fdf.empty:
            continue

        total_users = float(fdf["users"].sum())
        min_users = max(30.0, total_users * 0.005)
        candidates = fdf[fdf["users"] >= min_users].copy()
        if candidates.empty:
            candidates = fdf.copy()

        overall_conversion_rate = (
            float(fdf["converted_users"].sum()) / total_users
            if total_users > 0
            else 0.0
        )
        candidates["conversion_lift_pp"] = (
            pd.to_numeric(candidates["conversion_rate"], errors="coerce").fillna(0.0)
            - overall_conversion_rate
        ) * 100.0

        if direction == "low":
            selected = candidates.sort_values(["conversion_rate", "users"], ascending=[True, False]).head(1)
        else:
            selected = candidates.sort_values(["conversion_rate", "users"], ascending=[False, False]).head(1)
        if selected.empty:
            continue

        segment_row = selected.iloc[0]
        segment_label = demographic_report._classic_segment_label(feature_name, segment_row["segment_value"])
        rows.append(
            {
                "feature_name": feature_name,
                "feature_label": _business_feature_label(feature_name),
                "segment_label": str(segment_label),
                "feature_segment": f"{_business_feature_label(feature_name)} · {segment_label}",
                "users": float(segment_row.get("users", 0.0)),
                "conversion_rate": float(segment_row.get("conversion_rate", 0.0)),
                "conversion_lift_pp": float(segment_row.get("conversion_lift_pp", 0.0)),
                "mean_abs_shap": float(shap_row.get("mean_abs_shap") or 0.0),
                "mean_shap": float(shap_row.get("mean_shap") or 0.0),
                "importance_pct": float(shap_row.get("importance_pct") or 0.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _render_shap_segment_chart(segment_df: pd.DataFrame, *, title: str, color: str) -> None:
    if segment_df.empty:
        st.info(f"No {title.lower()} rows available.")
        return

    chart_df = segment_df.copy()
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
            y=alt.Y(
                "feature_segment:N",
                sort=alt.EncodingSortField(field="mean_abs_shap", op="sum", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=420, labelPadding=10),
            ),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("feature_name:N", title="Feature"),
                alt.Tooltip("segment_label:N", title="Observed segment"),
                alt.Tooltip("users:Q", title="Segment users", format=","),
                alt.Tooltip("conversion_rate:Q", title="Observed conversion", format=".1%"),
                alt.Tooltip("conversion_lift_pp:Q", title="Lift vs feature baseline (pp)", format=".1f"),
                alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".4f"),
                alt.Tooltip("mean_shap:Q", title="Mean SHAP", format=".4f"),
                alt.Tooltip("importance_pct:Q", title="Importance", format=".1%"),
            ],
        )
        .properties(title=title, height=max(180, min(430, 28 * len(chart_df))))
    )
    st.altair_chart(chart, use_container_width=True)


def _build_signal_table(stage_shap_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    if stage_shap_df.empty:
        return pd.DataFrame()

    low_segments = _build_observed_segment_rows(stage_shap_df, segments_df, direction="low")
    high_segments = _build_observed_segment_rows(stage_shap_df, segments_df, direction="high")
    low_lookup = {
        str(row["feature_name"]): row
        for _, row in low_segments.iterrows()
    }
    high_lookup = {
        str(row["feature_name"]): row
        for _, row in high_segments.iterrows()
    }

    rows: list[dict] = []
    for _, shap_row in stage_shap_df.iterrows():
        feature_name = str(shap_row.get("feature_name"))
        low = low_lookup.get(feature_name)
        high = high_lookup.get(feature_name)
        rows.append(
            {
                "business_signal": _business_feature_label(feature_name),
                "risk_segment": low.get("segment_label") if low is not None else "-",
                "risk_conversion": low.get("conversion_rate") if low is not None else None,
                "strong_segment": high.get("segment_label") if high is not None else "-",
                "strong_conversion": high.get("conversion_rate") if high is not None else None,
                "model_importance": shap_row.get("importance_pct"),
                "mean_shap": shap_row.get("mean_shap"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def _format_signal_table(signal_df: pd.DataFrame) -> pd.DataFrame:
    show = signal_df.copy()
    for column in ["risk_conversion", "strong_conversion", "model_importance"]:
        if column in show.columns:
            show[column] = show[column].apply(_fmt_pct)
    if "mean_shap" in show.columns:
        show["mean_shap"] = show["mean_shap"].apply(_fmt_float)
    return show


def _render_stage_signal_panel(stage: str, stage_df: pd.DataFrame, segments_df: pd.DataFrame) -> None:
    st.markdown(f"##### {JOURNEY_STAGE_LABELS[stage]}")
    st.caption(JOURNEY_STAGE_CAPTIONS[stage])

    if stage_df.empty:
        st.info("No top model signals mapped to this journey moment.")
        return

    signal_df = _build_signal_table(stage_df, segments_df)
    if signal_df.empty:
        st.info("No observed segment cuts available for these model signals.")
        return

    st.dataframe(
        _format_signal_table(signal_df),
        use_container_width=True,
        hide_index=True,
    )


def _render_shap_story(shap_df: pd.DataFrame, eval_df: pd.DataFrame, segments_df: pd.DataFrame) -> None:
    st.markdown("#### 3. What signals help explain the two friction points?")
    st.caption(
        "This is the business lens on SHAP. SHAP ranks which signals help the model identify likely conversion, "
        "while the segment columns show what actually happened in the cohort. This is not causal proof."
    )

    if shap_df.empty:
        st.warning("No SHAP dataset found.")
        return

    chart_df = shap_df.copy()
    chart_df["mean_abs_shap"] = pd.to_numeric(chart_df["mean_abs_shap"], errors="coerce").fillna(0.0)
    chart_df["mean_shap"] = pd.to_numeric(chart_df["mean_shap"], errors="coerce").fillna(0.0)
    chart_df = chart_df.sort_values(["mean_abs_shap", "feature_rank"], ascending=[False, True]).reset_index(drop=True)
    total_abs_shap = float(chart_df["mean_abs_shap"].sum())
    chart_df["importance_pct"] = chart_df["mean_abs_shap"] / total_abs_shap if total_abs_shap > 0 else 0.0

    if not eval_df.empty:
        eval_priority = eval_df.copy()
        eval_priority["split_priority"] = eval_priority["split_name"].astype(str).str.lower().map(
            {"test": 0, "validation": 1, "val": 1, "train": 2}
        ).fillna(99)
        best_eval = eval_priority.sort_values(["split_priority", "split_name"], ascending=[True, True]).iloc[0]
        st.markdown(
            " | ".join(
                [
                    f"**Model split:** `{str(best_eval.get('split_name', 'n/a')).lower()}`",
                    f"**ROC-AUC:** `{_fmt_pct(best_eval.get('roc_auc'))}`",
                    f"**PR-AUC:** `{_fmt_pct(best_eval.get('pr_auc'))}`",
                    f"**F1:** `{_fmt_pct(best_eval.get('f1'))}`",
                ]
            )
        )

    top_df = chart_df.head(DEFAULT_TOP_SHAP).copy()
    top_df["journey_stage"] = top_df["feature_name"].astype(str).apply(_feature_journey_stage)

    with st.expander("How are signals assigned to before vs after eKYC?", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "- **Before eKYC:** registration, login, device, channel, IP, contact, referrer, marketing, and source signals. These describe acquisition and early activation before the customer reaches verification.",
                    "- **After eKYC approval / deposit activation:** coupon, bonus, promotion, and pre-deposit incentive signals. These describe deposit activation before the first deposit; they are used as post-approval story signals only when interpreted together with the eKYC journey.",
                    "- **Customer profile:** age, nationality, country/region, and similar attributes. These describe the customer base and should not be interpreted as causal levers.",
                    "- This split is based on feature names/source semantics. It is not a separate SHAP model trained by journey step and should not be treated as causal evidence.",
                ]
            )
        )

    left, right = st.columns(2)
    with left:
        _render_stage_signal_panel("pre_ekyc", top_df[top_df["journey_stage"] == "pre_ekyc"].copy(), segments_df)
    with right:
        _render_stage_signal_panel(
            "post_approval",
            top_df[top_df["journey_stage"] == "post_approval"].copy(),
            segments_df,
        )

    lower_left, lower_right = st.columns(2)
    with lower_left:
        _render_stage_signal_panel(
            "customer_profile",
            top_df[top_df["journey_stage"] == "customer_profile"].copy(),
            segments_df,
        )
    with lower_right:
        _render_stage_signal_panel("other", top_df[top_df["journey_stage"] == "other"].copy(), segments_df)

    low_conversion_segments = _build_observed_segment_rows(top_df, segments_df, direction="low")
    high_conversion_segments = _build_observed_segment_rows(top_df, segments_df, direction="high")

    with st.expander("Model signal appendix: observed low/high conversion segments", expanded=False):
        left, right = st.columns(2)
        with left:
            _render_shap_segment_chart(
                low_conversion_segments,
                title="Observed low-conversion segments",
                color="#E76F51",
            )
        with right:
            _render_shap_segment_chart(
                high_conversion_segments,
                title="Observed high-conversion segments",
                color="#2EBD85",
            )

        show_df = top_df[
            ["feature_name", "journey_stage", "feature_rank", "mean_abs_shap", "mean_shap", "importance_pct", "sample_size"]
        ].copy()
        show_df["journey_stage"] = show_df["journey_stage"].map(JOURNEY_STAGE_LABELS).fillna(show_df["journey_stage"])
        show_df["importance_pct"] = show_df["importance_pct"].apply(_fmt_pct)
        show_df["mean_abs_shap"] = show_df["mean_abs_shap"].apply(_fmt_float)
        show_df["mean_shap"] = show_df["mean_shap"].apply(_fmt_float)
        st.dataframe(show_df, use_container_width=True, hide_index=True)


def _render_demographic_feature_story(segments_df: pd.DataFrame, feature_name: str) -> None:
    fdf = demographic_report._prepare_segments_feature_view(segments_df, feature_name)
    fdf = _apply_business_segment_filters(fdf, feature_name)
    if fdf.empty:
        return

    chart_df = fdf.sort_values(["segment_order", "users"], ascending=[True, False]).copy()
    chart_df["segment_label"] = chart_df["segment_value"].astype(str).apply(
        lambda value: demographic_report._classic_segment_label(feature_name, value)
    )
    segment_order = chart_df["segment_label"].tolist()
    total_users = float(chart_df["users"].sum())
    total_converted = float(chart_df["converted_users"].sum())
    overall_conversion_rate = total_converted / total_users if total_users > 0 else 0.0
    chart_df["conversion_lift_pp"] = (chart_df["conversion_rate"] - overall_conversion_rate) * 100.0

    top_non = chart_df.sort_values(["share_in_non_converted", "non_converted_users"], ascending=[False, False]).head(1)
    top_conv = chart_df.sort_values(["share_in_converted", "converted_users"], ascending=[False, False]).head(1)

    st.markdown(f"##### {_feature_label(feature_name)}")
    if not top_non.empty and not top_conv.empty:
        non_row = top_non.iloc[0]
        conv_row = top_conv.iloc[0]
        st.caption(
            f"Largest non-converted segment: `{non_row['segment_label']}` "
            f"({_fmt_pct(non_row.get('share_in_non_converted'))} of non-converted users). "
            f"Largest converted segment: `{conv_row['segment_label']}` "
            f"({_fmt_pct(conv_row.get('share_in_converted'))} of converted users)."
        )

    left, right = st.columns(2)
    with left:
        non_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "segment_label:N",
                    sort=segment_order,
                    title="Segment",
                    axis=alt.Axis(labelAngle=-35, labelLimit=120),
                ),
                y=alt.Y("share_in_non_converted:Q", title="Share within not converted", axis=alt.Axis(format="%")),
                color=alt.value("#E76F51"),
                tooltip=[
                    alt.Tooltip("segment_label:N", title="Segment"),
                    alt.Tooltip("non_converted_users:Q", title="Not converted users", format=",.0f"),
                    alt.Tooltip("share_in_non_converted:Q", title="Share within not converted", format=".1%"),
                    alt.Tooltip("conversion_rate:Q", title="Segment conversion", format=".1%"),
                    alt.Tooltip("population_pct:Q", title="Population share", format=".1%"),
                ],
            )
            .properties(title="Observed not converted", height=260)
        )
        st.altair_chart(non_chart, use_container_width=True)

    with right:
        conv_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "segment_label:N",
                    sort=segment_order,
                    title="Segment",
                    axis=alt.Axis(labelAngle=-35, labelLimit=120),
                ),
                y=alt.Y("share_in_converted:Q", title="Share within converted", axis=alt.Axis(format="%")),
                color=alt.value("#2EBD85"),
                tooltip=[
                    alt.Tooltip("segment_label:N", title="Segment"),
                    alt.Tooltip("converted_users:Q", title="Converted users", format=",.0f"),
                    alt.Tooltip("share_in_converted:Q", title="Share within converted", format=".1%"),
                    alt.Tooltip("conversion_rate:Q", title="Segment conversion", format=".1%"),
                    alt.Tooltip("population_pct:Q", title="Population share", format=".1%"),
                ],
            )
            .properties(title="Observed converted", height=260)
        )
        st.altair_chart(conv_chart, use_container_width=True)

    zero_rule = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
        color="#9AA4B2",
        strokeDash=[4, 4],
    ).encode(y="zero:Q")
    conversion_chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "segment_label:N",
                sort=segment_order,
                title="Segment",
                axis=alt.Axis(labelAngle=-35, labelLimit=140),
            ),
            y=alt.Y("conversion_lift_pp:Q", title="Observed conversion lift vs baseline (pp)"),
            color=alt.condition(
                "datum.conversion_lift_pp >= 0",
                alt.value("#2EBD85"),
                alt.value("#E76F51"),
            ),
            tooltip=[
                alt.Tooltip("segment_label:N", title="Segment"),
                alt.Tooltip("users:Q", title="Users", format=","),
                alt.Tooltip("conversion_rate:Q", title="Conversion rate", format=".1%"),
                alt.Tooltip("conversion_lift_pp:Q", title="Lift vs baseline (pp)", format=".1f"),
            ],
        )
        .properties(title="Observed conversion lift", height=380)
    )
    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        st.altair_chart(alt.layer(conversion_chart, zero_rule), use_container_width=True)


def _render_demographic_story(summary_df: pd.DataFrame, segments_df: pd.DataFrame) -> None:
    st.markdown("#### 4. Who are these customers?")
    st.caption(
        "This section is observed behavior/profile, not model inference. "
        "Use it to describe which customer segments over-index or under-index in conversion."
    )

    features = _available_demographic_features(summary_df, segments_df)
    if not features:
        st.warning("No demographic or registration descriptor segments found.")
        return

    for feature_name in features:
        _render_demographic_feature_story(segments_df, feature_name)


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
    analysis_key = f"funnel_final_story_analysis_select_{environment}"
    matured_key = f"funnel_final_story_matured_only_{environment}"
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
            help="Restricts outcome rates to customers with the complete observation window.",
        )

    try:
        _, shap_table, eval_table = overview_report._resolve_table_names(context, analysis_name)
        ekyc_table = ekyc_report._resolve_table_name(context, analysis_name)
        demographic_summary_table, demographic_segments_table = demographic_report._resolve_table_names(
            context,
            analysis_name,
        )

        shap_df = overview_report._load_shap(http_path, shap_table)
        eval_df = overview_report._load_eval(http_path, eval_table)
        summary_df = ekyc_report._load_summary(http_path, ekyc_table, matured_only)
        stage_df = ekyc_report._load_funnel_stages(http_path, ekyc_table, matured_only)
        flow_paths_df = ekyc_report._load_flow_paths(http_path, ekyc_table, matured_only)
        demographic_summary_df = demographic_report._load_summary(http_path, demographic_summary_table)
        demographic_segments_df = demographic_report._load_segments(http_path, demographic_segments_table)
    except Exception as exc:
        st.error(f"Failed to load final story datasets: {exc}")
        return

    if summary_df.empty:
        st.warning("No eKYC journey data found.")
        return

    try:
        freshness = load_report_freshness(http_path, ekyc_table)
    except Exception:
        freshness = {}

    row = summary_df.iloc[0]
    journey_window_days = int(row.get("journey_window_days") or 30)

    _render_business_css()

    st.subheader(f"Deposit Funnel Executive Story ({analysis_name})")
    st.caption(
        f"Business presentation view for the first {journey_window_days} days after registration. "
        "The report separates what actually happened from model hints, and frames the opportunity before eKYC and after eKYC approval."
    )
    render_report_freshness(freshness)

    _render_readout(row, flow_paths_df)

    st.divider()
    st.markdown("#### 1. Where does the funnel lose customers?")
    st.caption(
        "Observed customer counts by stage. This is the simplest way to show that the biggest losses are not inside eKYC approval itself."
    )
    ekyc_report._render_funnel_chart(stage_df)

    st.divider()
    st.markdown("#### 2. Which customer paths explain the loss?")
    st.caption(
        "Observed mutually exclusive paths. Use these rows to tell the story of healthy conversion, pre-eKYC drop-off, post-approval drop-off, and true eKYC friction."
    )
    ekyc_report._render_journey_flow_map(row, _filter_executive_flow_paths(flow_paths_df))

    st.divider()
    _render_shap_story(shap_df, eval_df, demographic_segments_df)

    st.divider()
    _render_demographic_story(demographic_summary_df, demographic_segments_df)
