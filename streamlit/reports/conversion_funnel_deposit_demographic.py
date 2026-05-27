from __future__ import annotations

import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select, resolve_sql_http_path

REPORT_META = {
    "title": "Deposit Conversion Funnel Demographics",
    "schema": "conversion_funnel_deposit",
    # "environments": ("training", "production"),
    "environments": ("training"),
    "description": "Side-by-side demographic characteristics for converted vs non-converted users.",
}

FUNNEL_ANALYSIS = os.getenv("FUNNEL_ANALYSIS", "analysis_2")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")

ANALYSIS_NAMES = {
    "analysis_1": "register_to_deposit_30d",
    "analysis_2": "register_to_deposit_30d_truewin",
    "analysis_3": "register_to_deposit_30d_igaming",
    "analysis_4": "register_to_deposit_30d_truewin_no_ekyc",
}

CLASSIC_FEATURES = [
    "registration_hour",
    "registration_weekday",
    "registration_day_of_month",
    "registration_weekend_flag",
    "registration_device",
    "days_since_registration",
    "customer_age_years",
    "customer_nationality",
]

FEATURE_NAME_ALIASES = {
    "registration_hour": "Registration Hour",
    "registration_weekday": "Registration Weekday",
    "registration_day_of_month": "Registration Day of Month",
    "registration_weekend_flag": "Weekend vs Weekday",
    "registration_device": "Registration Device",
    "days_since_registration": "Days Since Registration",
    "customer_age_years": "Customer Age (Years)",
    "customer_nationality": "Customer Nationality",
    "registered_by_mobile": "Registration Device",
    "registered_by_desktop": "Registration Device",
}

WEEKDAY_LABELS = {
    1: "sunday",
    2: "monday",
    3: "tuesday",
    4: "wednesday",
    5: "thursday",
    6: "friday",
    7: "saturday",
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


def _resolve_table_names(context: dict, analysis_name: str) -> tuple[str, str]:
    environment = str(context.get("environment", "production")).strip().lower()
    env_key = _environment_key(environment)

    default_catalog = "ml_train" if env_key == "TRAINING" else "ml_prod"
    default_summary = f"{default_catalog}.conversion_funnel_deposit.rpt_demographic_summary_{analysis_name}_current"
    default_segments = f"{default_catalog}.conversion_funnel_deposit.rpt_demographic_segments_{analysis_name}_current"

    summary_override = str(os.getenv(f"FUNNEL_DEMO_SUMMARY_TABLE_{env_key}", "")).strip() or str(os.getenv("FUNNEL_DEMO_SUMMARY_TABLE", "")).strip()
    segments_override = str(os.getenv(f"FUNNEL_DEMO_SEGMENTS_TABLE_{env_key}", "")).strip() or str(os.getenv("FUNNEL_DEMO_SEGMENTS_TABLE", "")).strip()

    summary_table = _safe_table_name(summary_override) if summary_override else _safe_table_name(default_summary)
    segments_table = _safe_table_name(segments_override) if segments_override else _safe_table_name(default_segments)
    return summary_table, segments_table


@st.cache_data(ttl=180)
def _load_summary(http_path: str, summary_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      feature_name,
      CAST(feature_rank AS INT) AS feature_rank,
      CAST(mean_abs_shap AS DOUBLE) AS mean_abs_shap,
      CAST(mean_shap AS DOUBLE) AS mean_shap,
      CAST(sample_size AS BIGINT) AS sample_size,
      CAST(n_users AS BIGINT) AS n_users,
      CAST(converted_users AS DOUBLE) AS converted_users,
      CAST(conversion_rate AS DOUBLE) AS conversion_rate,
      CAST(missing_users AS BIGINT) AS missing_users,
      CAST(missing_rate AS DOUBLE) AS missing_rate,
      CAST(approx_unique AS BIGINT) AS approx_unique,
      CAST(feature_mean AS DOUBLE) AS feature_mean,
      CAST(feature_median AS DOUBLE) AS feature_median,
      CAST(feature_std AS DOUBLE) AS feature_std
    FROM {summary_table}
    ORDER BY mean_abs_shap DESC, feature_rank ASC
    """
    return data_select(query=query, http_path=http_path)


@st.cache_data(ttl=180)
def _load_segments(http_path: str, segments_table: str) -> pd.DataFrame:
    query = f"""
    SELECT
      feature_name,
      CAST(feature_rank AS INT) AS feature_rank,
      CAST(mean_abs_shap AS DOUBLE) AS mean_abs_shap,
      CAST(mean_shap AS DOUBLE) AS mean_shap,
      CAST(sample_size AS BIGINT) AS sample_size,
      segment_kind,
      segment_value,
      CAST(segment_order AS DOUBLE) AS segment_order,
      CAST(users AS BIGINT) AS users,
      CAST(converted_users AS DOUBLE) AS converted_users,
      CAST(conversion_rate AS DOUBLE) AS conversion_rate,
      CAST(non_conversion_rate AS DOUBLE) AS non_conversion_rate,
      CAST(population_pct AS DOUBLE) AS population_pct,
      CAST(is_missing_segment AS INT) AS is_missing_segment
    FROM {segments_table}
    ORDER BY feature_rank ASC, segment_order ASC, users DESC
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


def _fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{int(digits)}f}"


def _to_int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _display_feature_name(feature_name: str) -> str:
    return FEATURE_NAME_ALIASES.get(feature_name, feature_name)


def _mobile_to_device_label(raw_value: str) -> str:
    value = str(raw_value)
    if value in {"__MISSING__", "__OTHER__", "__OUTSIDE__"}:
        return value
    ivalue = _to_int_or_none(value)
    if ivalue is None:
        return value
    return "Mobile" if ivalue == 1 else "Computer"


def _desktop_to_device_label(raw_value: str) -> str:
    value = str(raw_value)
    if value in {"__MISSING__", "__OTHER__", "__OUTSIDE__"}:
        return value
    ivalue = _to_int_or_none(value)
    if ivalue is None:
        return value
    return "Computer" if ivalue == 1 else "Mobile"


def _feature_source_name(segments_df: pd.DataFrame, feature_name: str) -> str:
    if feature_name != "registration_device":
        return feature_name
    available = set(segments_df["feature_name"].astype(str).unique().tolist())
    if "registered_by_mobile" in available:
        return "registered_by_mobile"
    if "registered_by_desktop" in available:
        return "registered_by_desktop"
    return feature_name


def _summary_source_name(summary_df: pd.DataFrame, feature_name: str) -> str:
    if feature_name != "registration_device":
        return feature_name
    available = set(summary_df["feature_name"].astype(str).unique().tolist())
    if "registered_by_mobile" in available:
        return "registered_by_mobile"
    if "registered_by_desktop" in available:
        return "registered_by_desktop"
    return feature_name


def _classic_segment_label(feature_name: str, raw_value: str) -> str:
    value = str(raw_value)
    if value in {"__MISSING__", "__OTHER__", "__OUTSIDE__"}:
        return value

    ivalue = _to_int_or_none(value)
    if ivalue is None:
        return value

    if feature_name == "registration_weekday":
        return WEEKDAY_LABELS.get(ivalue, str(ivalue))
    if feature_name == "registration_weekend_flag":
        return "Weekend" if ivalue == 1 else "Weekday"
    if feature_name == "registration_device":
        return "Mobile" if ivalue == 1 else "Computer"
    if feature_name in {"registration_hour", "registration_day_of_month", "days_since_registration"}:
        return str(ivalue)
    if feature_name == "customer_nationality" and value == "UNKNOWN":
        return "Unknown"
    return str(ivalue)


def _classic_feature_chart_type(feature_name: str) -> str:
    if feature_name in {"registration_hour", "registration_weekday", "registration_day_of_month"}:
        return "calendar_discrete"
    if feature_name in {"registration_weekend_flag", "registration_device"}:
        return "binary_flag"
    return "binned_numeric"


def _prepare_segments_feature_view(segments_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    source_name = _feature_source_name(segments_df, feature_name)
    fdf = segments_df[segments_df["feature_name"] == source_name].copy()
    if fdf.empty:
        return fdf

    if feature_name == "registration_device":
        if source_name == "registered_by_mobile":
            fdf["segment_value"] = fdf["segment_value"].astype(str).apply(_mobile_to_device_label)
            fdf["segment_order"] = fdf["segment_value"].map({"Mobile": 1.0, "Computer": 2.0}).fillna(fdf["segment_order"])
        elif source_name == "registered_by_desktop":
            fdf["segment_value"] = fdf["segment_value"].astype(str).apply(_desktop_to_device_label)
            fdf["segment_order"] = fdf["segment_value"].map({"Mobile": 1.0, "Computer": 2.0}).fillna(fdf["segment_order"])

    fdf["users"] = pd.to_numeric(fdf["users"], errors="coerce").fillna(0.0)
    fdf["converted_users"] = pd.to_numeric(fdf["converted_users"], errors="coerce").fillna(0.0)
    fdf["conversion_rate"] = pd.to_numeric(fdf["conversion_rate"], errors="coerce").fillna(0.0)
    fdf["population_pct"] = pd.to_numeric(fdf["population_pct"], errors="coerce").fillna(0.0)
    fdf["segment_order"] = pd.to_numeric(fdf["segment_order"], errors="coerce").fillna(999999.0)
    fdf["non_converted_users"] = (fdf["users"] - fdf["converted_users"]).clip(lower=0.0)

    total_converted = float(fdf["converted_users"].sum())
    total_non_converted = float(fdf["non_converted_users"].sum())
    fdf["share_in_converted"] = fdf["converted_users"] / total_converted if total_converted > 0 else 0.0
    fdf["share_in_non_converted"] = fdf["non_converted_users"] / total_non_converted if total_non_converted > 0 else 0.0
    return fdf


def _build_class_characteristics_table(segments_df: pd.DataFrame, features: list[str], class_side: str) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_name in features:
        fdf = _prepare_segments_feature_view(segments_df, feature_name)
        if fdf.empty:
            continue

        if class_side == "converted":
            candidate = fdf[fdf["converted_users"] > 0].sort_values(["share_in_converted", "converted_users"], ascending=[False, False]).head(1)
            if candidate.empty:
                continue
            row = candidate.iloc[0]
            rows.append(
                {
                    "feature_name": _display_feature_name(feature_name),
                    "main_segment": _classic_segment_label(feature_name, row["segment_value"]),
                    "users": float(row["converted_users"]),
                    "class_share": float(row["share_in_converted"]),
                    "conversion_rate": float(row["conversion_rate"]),
                    "population_pct": float(row["population_pct"]),
                }
            )
        else:
            candidate = fdf[fdf["non_converted_users"] > 0].sort_values(["share_in_non_converted", "non_converted_users"], ascending=[False, False]).head(1)
            if candidate.empty:
                continue
            row = candidate.iloc[0]
            rows.append(
                {
                    "feature_name": _display_feature_name(feature_name),
                    "main_segment": _classic_segment_label(feature_name, row["segment_value"]),
                    "users": float(row["non_converted_users"]),
                    "class_share": float(row["share_in_non_converted"]),
                    "conversion_rate": float(row["conversion_rate"]),
                    "population_pct": float(row["population_pct"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("class_share", ascending=False).reset_index(drop=True)


def _feature_summary_row(summary_df: pd.DataFrame, feature_name: str) -> pd.Series | None:
    source_name = _summary_source_name(summary_df, feature_name)
    row = summary_df[summary_df["feature_name"] == source_name].head(1)
    if row.empty:
        return None
    return row.iloc[0]


def _render_feature_detail_panel(segments_df: pd.DataFrame, feature_name: str) -> None:
    fdf = _prepare_segments_feature_view(segments_df, feature_name)
    if fdf.empty:
        st.info("No segment rows found for selected feature.")
        return

    chart_df = fdf.sort_values(["segment_order", "users"], ascending=[True, False]).copy()
    chart_df["segment_label"] = chart_df["segment_value"].astype(str).apply(lambda x: _classic_segment_label(feature_name, x))
    sort_order = chart_df["segment_label"].tolist()
    total_users = float(chart_df["users"].sum())
    total_converted = float(chart_df["converted_users"].sum())
    overall_conversion_rate = (total_converted / total_users) if total_users > 0 else 0.0
    chart_df["conversion_lift_pp"] = (chart_df["conversion_rate"] - overall_conversion_rate) * 100.0

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("###### Not Converted")
        non_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("segment_label:N", sort=sort_order, title="Segment"),
                y=alt.Y("share_in_non_converted:Q", title="Share within not converted"),
                tooltip=[
                    "segment_label",
                    alt.Tooltip("non_converted_users:Q", title="non_converted_users", format=",.0f"),
                    alt.Tooltip("users:Q", title="segment_users", format=",.0f"),
                    alt.Tooltip("share_in_non_converted:Q", title="share_within_non_converted", format=".2%"),
                    alt.Tooltip("conversion_rate:Q", title="segment_conversion_rate", format=".2%"),
                    alt.Tooltip("population_pct:Q", format=".2%"),
                ],
            )
        )
        st.altair_chart(non_chart, use_container_width=True)

    with right_col:
        st.markdown("###### Converted")
        conv_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("segment_label:N", sort=sort_order, title="Segment"),
                y=alt.Y("share_in_converted:Q", title="Share within converted"),
                tooltip=[
                    "segment_label",
                    alt.Tooltip("converted_users:Q", title="converted_users", format=",.0f"),
                    alt.Tooltip("users:Q", title="segment_users", format=",.0f"),
                    alt.Tooltip("share_in_converted:Q", title="share_within_converted", format=".2%"),
                    alt.Tooltip("conversion_rate:Q", title="segment_conversion_rate", format=".2%"),
                    alt.Tooltip("population_pct:Q", format=".2%"),
                ],
            )
        )
        st.altair_chart(conv_chart, use_container_width=True)

    st.markdown("###### Normalized comparison")
    st.caption(
        "Positive values mean segment conversion is above overall baseline."
    )
    pad_left, center_col, pad_right = st.columns([1, 2, 1])
    with center_col:
        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9AA4B2").encode(y="y:Q")
        lift_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("segment_label:N", sort=sort_order, title="Segment"),
                y=alt.Y("conversion_lift_pp:Q", title="Conversion lift vs overall (pp)"),
                color=alt.condition(
                    "datum.conversion_lift_pp >= 0",
                    alt.value("#2EBD85"),
                    alt.value("#E76F51"),
                ),
                tooltip=[
                    "segment_label",
                    alt.Tooltip("conversion_rate:Q", title="segment_conversion_rate", format=".2%"),
                    alt.Tooltip("conversion_lift_pp:Q", title="lift_pp", format=".2f"),
                    alt.Tooltip("users:Q", format=","),
                    alt.Tooltip("population_pct:Q", format=".2%"),
                ],
            )
        )
        st.altair_chart(alt.layer(zero_rule, lift_chart), use_container_width=True)

    st.markdown("###### Segment conversion")
    st.caption(f"Overall conversion baseline for this feature: {_fmt_pct(overall_conversion_rate)}")
    conv_rate_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("segment_label:N", sort=sort_order, title="Segment"),
            y=alt.Y("conversion_rate:Q", title="Conversion rate"),
            tooltip=[
                "segment_label",
                alt.Tooltip("users:Q", format=","),
                alt.Tooltip("converted_users:Q", format=",.0f"),
                alt.Tooltip("conversion_rate:Q", format=".2%"),
                alt.Tooltip("population_pct:Q", format=".2%"),
            ],
        )
    )
    st.altair_chart(conv_rate_chart, use_container_width=True)


def _render_classic_feature_cards(segments_df: pd.DataFrame, features: list[str], summary_df: pd.DataFrame) -> None:
    if not features:
        st.info("No classic features available.")
        return

    for feature_name in features:
        fdf = _prepare_segments_feature_view(segments_df, feature_name)
        if fdf.empty:
            continue

        row = _feature_summary_row(summary_df, feature_name)
        title = _display_feature_name(feature_name)
        with st.expander(title, expanded=False):
            if row is not None:
                st.caption(
                    f"mean_abs_shap={_fmt_float(row.get('mean_abs_shap'))} | "
                    f"mean_shap={_fmt_float(row.get('mean_shap'))} | "
                    f"missing_rate={_fmt_pct(row.get('missing_rate'))}"
                )
            _render_feature_detail_panel(segments_df, feature_name)


def _render_section(title: str, features: list[str], segments_df: pd.DataFrame, summary_df: pd.DataFrame, key_prefix: str, enable_feature_selector: bool = True) -> None:
    st.markdown(f"##### {title}")
    if not features:
        st.info("No features available in this section.")
        return

    left_col, right_col = st.columns(2)
    non_conv_summary = _build_class_characteristics_table(segments_df, features, class_side="non_converted")
    conv_summary = _build_class_characteristics_table(segments_df, features, class_side="converted")

    with left_col:
        st.markdown("###### Not Converted - Main characteristics")
        if non_conv_summary.empty:
            st.info("No non-converted profile found.")
        else:
            show = non_conv_summary.copy()
            show["users"] = show["users"].apply(_fmt_num)
            show["class_share"] = show["class_share"].apply(_fmt_pct)
            show["conversion_rate"] = show["conversion_rate"].apply(_fmt_pct)
            show["population_pct"] = show["population_pct"].apply(_fmt_pct)
            st.dataframe(
                show.rename(
                    columns={
                        "main_segment": "dominant_segment",
                        "users": "non_converted_users",
                        "class_share": "share_within_non_converted",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with right_col:
        st.markdown("###### Converted - Main characteristics")
        if conv_summary.empty:
            st.info("No converted profile found.")
        else:
            show = conv_summary.copy()
            show["users"] = show["users"].apply(_fmt_num)
            show["class_share"] = show["class_share"].apply(_fmt_pct)
            show["conversion_rate"] = show["conversion_rate"].apply(_fmt_pct)
            show["population_pct"] = show["population_pct"].apply(_fmt_pct)
            st.dataframe(
                show.rename(
                    columns={
                        "main_segment": "dominant_segment",
                        "users": "converted_users",
                        "class_share": "share_within_converted",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    if enable_feature_selector:
        selected_feature = st.selectbox(
            f"{title} - Feature detail",
            options=features,
            index=0,
            format_func=_display_feature_name,
            key=f"{key_prefix}_feature_detail",
        )

        detail_row = _feature_summary_row(summary_df, selected_feature)
        if detail_row is not None:
            st.caption(
                f"Feature={_display_feature_name(selected_feature)} | mean_abs_shap={_fmt_float(detail_row.get('mean_abs_shap'))} | "
                f"mean_shap={_fmt_float(detail_row.get('mean_shap'))} | missing_rate={_fmt_pct(detail_row.get('missing_rate'))}"
            )
        _render_feature_detail_panel(segments_df, selected_feature)
    else:
        _render_classic_feature_cards(segments_df=segments_df, features=features, summary_df=summary_df)


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
        f"funnel_demographic_analysis_select_{str(context.get('environment', 'other')).strip().lower()}"
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
        summary_table, segments_table = _resolve_table_names(context, analysis_name)
        summary_df = _load_summary(http_path, summary_table)
        segments_df = _load_segments(http_path, segments_table)
    except Exception as exc:
        st.error(f"Failed to load demographic datasets: {exc}")
        return

    if summary_df.empty or segments_df.empty:
        st.warning("No demographic datasets found.")
        return

    summary_df = summary_df.sort_values(["mean_abs_shap", "feature_rank"], ascending=[False, True]).reset_index(drop=True)
    segments_df = segments_df.copy()

    st.subheader(f"Demographic Deep Dive ({analysis_name})")
    st.caption(
        "Left side = not converted users, right side = converted users. "
        "Classical features are shown as detailed cards by feature."
    )

    overall_n = None
    overall_conv_rate = None
    if not summary_df.empty:
        try:
            overall_n = int(pd.to_numeric(summary_df["n_users"], errors="coerce").dropna().max())
        except Exception:
            overall_n = None
        try:
            overall_conv_rate = float(pd.to_numeric(summary_df["conversion_rate"], errors="coerce").dropna().max())
        except Exception:
            overall_conv_rate = None
    if overall_n is not None and overall_conv_rate is not None:
        st.markdown(
            f"- Overall cohort in this analysis: `n={_fmt_num(overall_n)}` | "
            f"`conversion={_fmt_pct(overall_conv_rate)}`"
        )

    available_features = set(summary_df["feature_name"].astype(str).tolist())
    classic_features: list[str] = []
    for feature_name in CLASSIC_FEATURES:
        if feature_name == "registration_device":
            if ("registered_by_mobile" in available_features) or ("registered_by_desktop" in available_features):
                classic_features.append(feature_name)
        elif feature_name in available_features:
            classic_features.append(feature_name)

    shap_ranked = (
        summary_df[summary_df["feature_rank"].notna()]
        .sort_values(["feature_rank", "mean_abs_shap"], ascending=[True, False])
        .copy()
    )
    max_top_n = min(30, max(1, len(shap_ranked)))
    default_top_n = min(12, max_top_n)
    top_n = st.slider("Top SHAP features to include", min_value=1, max_value=max_top_n, value=default_top_n)
    shap_features = shap_ranked.head(top_n)["feature_name"].astype(str).tolist()

    _render_section(
        title="Classical Features",
        features=classic_features,
        segments_df=segments_df,
        summary_df=summary_df,
        key_prefix="classic",
        enable_feature_selector=False,
    )

    _render_section(
        title="Top SHAP Features",
        features=shap_features,
        segments_df=segments_df,
        summary_df=summary_df,
        key_prefix="shap",
    )
