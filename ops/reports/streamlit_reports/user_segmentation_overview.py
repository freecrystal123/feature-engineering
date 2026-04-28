from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


def _sample_frame() -> pd.DataFrame:
    rows = [
        ("2026-01-01", "High Value", 920, 0.84, 612.5),
        ("2026-01-01", "Growth", 1840, 0.66, 252.7),
        ("2026-01-01", "At Risk", 1210, 0.41, 118.3),
        ("2026-02-01", "High Value", 947, 0.85, 618.9),
        ("2026-02-01", "Growth", 1899, 0.67, 259.2),
        ("2026-02-01", "At Risk", 1170, 0.40, 115.1),
        ("2026-03-01", "High Value", 971, 0.86, 624.8),
        ("2026-03-01", "Growth", 1941, 0.68, 267.6),
        ("2026-03-01", "At Risk", 1123, 0.39, 111.2),
        ("2026-04-01", "High Value", 1004, 0.87, 632.1),
        ("2026-04-01", "Growth", 2018, 0.69, 273.9),
        ("2026-04-01", "At Risk", 1070, 0.38, 109.4),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["month", "segment", "users", "retention_30d", "avg_monthly_wager"],
    )
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


def render(context: dict | None = None) -> None:
    context = context or {}
    st.subheader("User Segmentation Overview")
    st.caption(
        f"Environment: {context.get('environment', 'n/a')} | "
        f"Analysis: {context.get('schema', 'user_segmentation')}"
    )

    df = _sample_frame()
    available_segments = sorted(df["segment"].unique().tolist())
    selected_segments = st.multiselect(
        "Segments",
        options=available_segments,
        default=available_segments,
    )

    month_min = df["month"].min().date()
    month_max = df["month"].max().date()
    selected_range = st.slider(
        "Month range",
        min_value=month_min,
        max_value=month_max,
        value=(month_min, month_max),
    )

    filtered = df[
        (df["segment"].isin(selected_segments))
        & (df["month"].dt.date >= selected_range[0])
        & (df["month"].dt.date <= selected_range[1])
    ].copy()

    if filtered.empty:
        st.info("No data for the selected filters.")
        return

    total_users = int(filtered["users"].sum())
    avg_retention = float(filtered["retention_30d"].mean()) * 100
    avg_wager = float(filtered["avg_monthly_wager"].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("Users in scope", f"{total_users:,}")
    col2.metric("Average retention (30d)", f"{avg_retention:.1f}%")
    col3.metric("Average monthly wager", f"{avg_wager:.1f}")

    st.markdown("##### Retention trend by segment")
    trend = filtered.pivot_table(
        index="month",
        columns="segment",
        values="retention_30d",
        aggfunc="mean",
    )
    st.line_chart(trend, use_container_width=True)

    table = filtered.copy()
    table["month"] = table["month"].dt.date
    table["retention_30d"] = (table["retention_30d"] * 100).round(1).astype(str) + "%"
    st.dataframe(table, use_container_width=True, hide_index=True)

