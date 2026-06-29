from __future__ import annotations

import html
import re
from datetime import date, datetime

import pandas as pd
import streamlit as st

from core.databricks_query_service import data_select

TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")


def _safe_table_name(name: str) -> str:
    table_name = str(name).strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table identifier: {table_name}")
    return table_name


def _value_or_none(value):
    if value is None or pd.isna(value):
        return None
    return value


@st.cache_data(ttl=180, show_spinner="Loading...")
def load_report_freshness(http_path: str, table_name: str) -> dict:
    safe_table_name = _safe_table_name(table_name)
    query = f"""
    SELECT
      MAX(CAST(ref_date AS DATE)) AS data_as_of,
      MAX(CAST(created_at AS TIMESTAMP)) AS report_updated_at
    FROM {safe_table_name}
    """
    metadata_df = data_select(query=query, http_path=http_path)
    if metadata_df.empty:
        return {"data_as_of": None, "report_updated_at": None}

    row = metadata_df.iloc[0]
    return {
        "data_as_of": _value_or_none(row.get("data_as_of")),
        "report_updated_at": _value_or_none(row.get("report_updated_at")),
    }


def _fmt_date(value) -> str:
    normalized = _value_or_none(value)
    if normalized is None:
        return "Unavailable"
    if isinstance(normalized, datetime):
        return normalized.strftime("%d %b %Y")
    if isinstance(normalized, date):
        return normalized.strftime("%d %b %Y")
    try:
        return pd.to_datetime(normalized).strftime("%d %b %Y")
    except Exception:
        return str(normalized)


def _fmt_datetime(value) -> str:
    normalized = _value_or_none(value)
    if normalized is None:
        return "Unavailable"
    if isinstance(normalized, datetime):
        return normalized.strftime("%d %b %Y · %H:%M")
    try:
        return pd.to_datetime(normalized).strftime("%d %b %Y · %H:%M")
    except Exception:
        return str(normalized)


def render_report_freshness(metadata: dict) -> None:
    data_as_of = html.escape(_fmt_date(metadata.get("data_as_of")))
    report_updated_at = html.escape(_fmt_datetime(metadata.get("report_updated_at")))
    st.markdown(
        f"""
        <div class="report-freshness">
          <div class="report-freshness-item">
            <span class="report-freshness-dot"></span>
            <div>
              <span class="report-freshness-label">Data as of</span>
              <strong>{data_as_of}</strong>
            </div>
          </div>
          <div class="report-freshness-item">
            <span class="report-freshness-dot report-freshness-dot-accent"></span>
            <div>
              <span class="report-freshness-label">Report updated</span>
              <strong>{report_updated_at}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
