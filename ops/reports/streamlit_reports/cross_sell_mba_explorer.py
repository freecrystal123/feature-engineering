from __future__ import annotations

import pandas as pd
import streamlit as st


def _sample_rules() -> pd.DataFrame:
    rows = [
        ("casino::speed baccarat i", "sports::basketball", 3.15, 0.184, 112),
        ("sports::football", "casino::roulette", 2.42, 0.142, 209),
        ("casino::blackjack", "sports::tennis", 2.01, 0.118, 76),
        ("casino::roulette", "sports::soccer", 1.78, 0.096, 184),
        ("sports::basketball", "casino::slot premium", 1.65, 0.089, 137),
    ]
    return pd.DataFrame(
        rows,
        columns=["antecedent", "consequent", "lift", "confidence", "users_in_both"],
    )


def render(context: dict | None = None) -> None:
    context = context or {}
    st.subheader("Cross-sell MBA Explorer")
    st.caption(
        f"Environment: {context.get('environment', 'n/a')} | "
        f"Analysis: {context.get('schema', 'cross_sell_mba')}"
    )

    rules = _sample_rules()
    min_lift = st.slider("Minimum lift", min_value=1.0, max_value=4.0, value=1.5, step=0.05)
    min_confidence = st.slider(
        "Minimum confidence (%)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )

    filtered = rules[
        (rules["lift"] >= min_lift)
        & (rules["confidence"] * 100 >= min_confidence)
    ].copy()

    if filtered.empty:
        st.warning("No rules for the selected thresholds.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Rules", f"{len(filtered):,}")
    col2.metric("Average lift", f"{filtered['lift'].mean():.2f}x")
    col3.metric("Users in both (sum)", f"{int(filtered['users_in_both'].sum()):,}")

    display = filtered.copy()
    display["confidence"] = (display["confidence"] * 100).round(2).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("##### Lift by antecedent")
    st.bar_chart(filtered.set_index("antecedent")[["lift"]], use_container_width=True)

