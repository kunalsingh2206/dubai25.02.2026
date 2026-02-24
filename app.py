from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.etl import etl_run, load_processed
from src.metrics import Filters, apply_filters, compute_kpis


st.set_page_config(
    page_title="Dubai Real Estate Transactions | Executive Dashboard",
    page_icon="🏙️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      .sobha-accent { color: #1e88e5; font-weight: 700; }
      .small-note { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).parent
RAW_XLSX = ROOT / "data" / "raw" / "transactions.xlsx"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "transactions_clean.parquet"
PROCESSED_PKL = ROOT / "data" / "processed" / "transactions_clean.pkl"
ALIAS_CSV = ROOT / "config" / "developer_aliases.csv"

KPI_COLUMNS = [
    "Transactions",
    "Transaction%",
    "Sales Velocity (txn/day)",
    "Total Sales Amount (AED)",
    "Sales Amount (AED) %",
    "Avg Selling Price (AED)",
    "Median Selling Price (AED)",
    "Avg Size (Sqf)",
    "Median Size (Sqf)",
    "Mode Size (Sqf)",
    "Avg PSF (AED/Sqf)",
    "Median PSF (AED/Sqf)",
    "Mode PSF (AED/Sqf)",
]

TICKET_BUCKET_ORDER = [
    "<500K",
    "500K-700K",
    "700K-900K",
    "900K-1.2M",
    "1.2M-1.5M",
    "1.5M-2M",
    "2M-2.5M",
    "2.5M-3M",
    "3M-4M",
    "4M-5M",
    "5M-7M",
    "7M-10M",
    "10M+",
]

PSF_BUCKET_ORDER = ["<1,500", "1,500-2,000", "2,000-2,500", "2,500-3,000", "3,000+"]


def _processed_path_preference() -> Path:
    try:
        import pyarrow  # noqa: F401

        return PROCESSED_PARQUET
    except Exception:
        return PROCESSED_PKL


@st.cache_data(show_spinner=False)
def load_data(force_rebuild: bool = False) -> Tuple[pd.DataFrame, Dict]:
    processed_path = _processed_path_preference()
    processed_exists = processed_path.exists()
    raw_mtime = RAW_XLSX.stat().st_mtime if RAW_XLSX.exists() else 0
    processed_mtime = processed_path.stat().st_mtime if processed_exists else 0
    needs_build = force_rebuild or (not processed_exists) or (raw_mtime > processed_mtime)

    audit = {}
    if needs_build:
        audit = etl_run(
            raw_xlsx_path=str(RAW_XLSX),
            out_path=str(processed_path),
            developer_alias_csv=str(ALIAS_CSV) if ALIAS_CSV.exists() else None,
        )
    return load_processed(str(processed_path)), audit


def is_missing(x: object) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def fmt_whole(x: float | int | None) -> str:
    if is_missing(x):
        return "—"
    return f"{int(round(float(x))):,}"


def fmt_pct(x: float | int | None) -> str:
    if is_missing(x):
        return "—"
    return f"{float(x):,.2f}%"


def fmt_aed_compact(x: float | int | None) -> str:
    if is_missing(x):
        return "—"
    v = float(x)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000_000:
        return f"{sign}AED {a / 1_000_000_000:.2f} Bn"
    if a >= 1_000_000:
        return f"{sign}AED {a / 1_000_000:.2f} Mn"
    return f"{sign}AED {a:,.0f}"


def formatted_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"Scenario", "Rank", C.COL_COMMUNITY, "Developer Group", C.COL_PROPERTY}:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m")
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in {"Transaction%", "Sales Amount (AED) %"}:
            out[col] = out[col].apply(fmt_pct)
        elif "(AED)" in col and "PSF" not in col:
            out[col] = out[col].apply(fmt_aed_compact)
        else:
            out[col] = out[col].apply(fmt_whole)
    return out


def available_granularities(span_days: int) -> List[str]:
    if span_days <= 7:
        return ["Day", "Week"]
    if span_days <= 31:
        return ["Day", "Week"]
    if span_days <= 365:
        return ["Day", "Week", "Month"]
    return ["Day", "Week", "Month", "Quarter", "Year"]


def aggregate_series(df: pd.DataFrame, basis: str, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Date": [], "Value": []})

    d = df.copy()
    d["Date"] = pd.to_datetime(d[C.COL_DATE]).dt.normalize()
    d = d[d["Date"].dt.dayofweek < 5].copy()
    if d.empty:
        return pd.DataFrame({"Date": [], "Value": []})

    if basis == "amount":
        d["Value"] = d[C.COL_AMOUNT_AED].astype(float)
    else:
        d["Value"] = 1.0

    if granularity == "Day":
        g = d.groupby("Date", as_index=False)["Value"].sum()
        all_days = pd.date_range(g["Date"].min(), g["Date"].max(), freq="B")
        g = g.set_index("Date").reindex(all_days, fill_value=0).rename_axis("Date").reset_index()
        return g

    if granularity == "Week":
        # Use Monday-start weeks.
        d["Period"] = d["Date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    elif granularity == "Month":
        d["Period"] = d["Date"].dt.to_period("M").dt.start_time
    elif granularity == "Quarter":
        d["Period"] = d["Date"].dt.to_period("Q").dt.start_time
    else:
        d["Period"] = d["Date"].dt.to_period("Y").dt.start_time

    g = d.groupby("Period", as_index=False)["Value"].sum().rename(columns={"Period": "Date"})
    return g.sort_values("Date").reset_index(drop=True)


def cumulative_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Date": [], "Value": []})
    out = df.sort_values("Date").copy()
    out["Value"] = out["Value"].cumsum()
    return out


def safe_options(series: pd.Series) -> List[str]:
    return [str(x) for x in series.dropna().astype(str).unique().tolist()]


def numeric_bounds(df: pd.DataFrame, col: str, fallback_df: pd.DataFrame) -> Tuple[float, float]:
    src = df if not df.empty else fallback_df
    lo = float(src[col].min())
    hi = float(src[col].max())
    if np.isnan(lo) or np.isnan(hi) or lo == hi:
        lo, hi = float(fallback_df[col].min()), float(fallback_df[col].max())
    return lo, hi


def scenario_filter_widgets(prefix: str, label_default: str, df: pd.DataFrame) -> Tuple[str, Filters]:
    latest_date = pd.to_datetime(df[C.COL_DATE]).max().date()
    earliest_date = pd.to_datetime(df[C.COL_DATE]).min().date()

    scenario_name = st.text_input("Scenario name", value=label_default, key=f"{prefix}_name")
    preset = st.radio(
        "Date preset",
        ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
        horizontal=True,
        key=f"{prefix}_date_preset",
    )
    if preset == "Last 7 days":
        dmin = latest_date - pd.Timedelta(days=6)
        dmax = latest_date
    elif preset == "Last 30 days":
        dmin = latest_date - pd.Timedelta(days=29)
        dmax = latest_date
    elif preset == "Last 90 days":
        dmin = latest_date - pd.Timedelta(days=89)
        dmax = latest_date
    else:
        dmin, dmax = st.date_input(
            "Date range",
            value=(earliest_date, latest_date),
            min_value=earliest_date,
            max_value=latest_date,
            key=f"{prefix}_date_range",
        )

    scoped = df[
        (pd.to_datetime(df[C.COL_DATE]) >= pd.to_datetime(dmin))
        & (pd.to_datetime(df[C.COL_DATE]) <= pd.to_datetime(dmax))
    ].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        tx_options = ["Combined"] + sorted(safe_options(scoped["Transaction Bucket"]))
        tx_bucket = st.selectbox("Transaction Type", options=tx_options, key=f"{prefix}_tx_bucket")
    if tx_bucket != "Combined":
        scoped = scoped[scoped["Transaction Bucket"] == tx_bucket]

    with col2:
        prop_options = ["Combined"] + sorted(safe_options(scoped[C.COL_PROPERTY_TYPE]))
        prop_type = st.selectbox("Type", options=prop_options, key=f"{prefix}_prop_type")
    if prop_type != "Combined":
        scoped = scoped[scoped[C.COL_PROPERTY_TYPE] == prop_type]

    with col3:
        include_outliers = st.toggle("Include outliers", value=False, key=f"{prefix}_outliers")
    if (not include_outliers) and ("Any Outlier" in scoped.columns):
        scoped = scoped[~scoped["Any Outlier"]]

    allowed = [b for b in C.ALLOWED_BEDROOMS if b in set(safe_options(scoped[C.COL_BEDROOMS]))]
    bedrooms = st.multiselect("Bedrooms", options=allowed, default=allowed, key=f"{prefix}_bedrooms")
    if bedrooms:
        scoped = scoped[scoped[C.COL_BEDROOMS].isin(bedrooms)]

    s_min, s_max = numeric_bounds(scoped, C.COL_SIZE_SQF, df)
    a_min, a_max = numeric_bounds(scoped, C.COL_AMOUNT_AED, df)
    p_min, p_max = numeric_bounds(scoped, C.COL_AED_PSF, df)

    r1, r2, r3 = st.columns(3)
    with r1:
        size_range = st.slider(
            "Size (Sqf)",
            min_value=float(s_min),
            max_value=float(s_max),
            value=(float(s_min), float(s_max)),
            step=1.0,
            key=f"{prefix}_size_range",
        )
    with r2:
        amount_range = st.slider(
            "Amount (AED)",
            min_value=float(a_min),
            max_value=float(a_max),
            value=(float(a_min), float(a_max)),
            step=float(max(1000.0, (a_max - a_min) / 2000.0)),
            key=f"{prefix}_amount_range",
        )
    with r3:
        psf_range = st.slider(
            "AED/Sqf",
            min_value=float(p_min),
            max_value=float(p_max),
            value=(float(p_min), float(p_max)),
            step=float(max(1.0, (p_max - p_min) / 2000.0)),
            key=f"{prefix}_psf_range",
        )

    scoped = scoped[
        (scoped[C.COL_SIZE_SQF] >= size_range[0])
        & (scoped[C.COL_SIZE_SQF] <= size_range[1])
        & (scoped[C.COL_AMOUNT_AED] >= amount_range[0])
        & (scoped[C.COL_AMOUNT_AED] <= amount_range[1])
        & (scoped[C.COL_AED_PSF] >= psf_range[0])
        & (scoped[C.COL_AED_PSF] <= psf_range[1])
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        community_options = scoped[C.COL_COMMUNITY].value_counts().index.tolist()
        community = st.multiselect("Community (optional)", options=community_options, default=[], key=f"{prefix}_community")
    if community:
        scoped = scoped[scoped[C.COL_COMMUNITY].isin(community)]

    with c2:
        dev_options = scoped["Developer Group"].value_counts().index.tolist()
        developer = st.multiselect("Developer (optional)", options=dev_options, default=[], key=f"{prefix}_developer")
    if developer:
        scoped = scoped[scoped["Developer Group"].isin(developer)]

    with c3:
        prop_options = scoped[C.COL_PROPERTY].value_counts().index.tolist()
        prop = st.multiselect("Property / Project (optional)", options=prop_options, default=[], key=f"{prefix}_property")

    return scenario_name.strip() or label_default, Filters(
        date_min=pd.to_datetime(dmin),
        date_max=pd.to_datetime(dmax),
        transaction_bucket=tx_bucket,
        property_type=prop_type,
        bedrooms=bedrooms,
        community=community or None,
        developer_group=developer or None,
        prop=prop or None,
        size_range=size_range,
        amount_range=amount_range,
        psf_range=psf_range,
        include_outliers=include_outliers,
    )


def scenario_kpi_rows(results: Sequence[Dict]) -> pd.DataFrame:
    total_tx = sum(int(r["kpis"]["Transactions"]) for r in results)
    total_amt = sum(float(r["kpis"]["Total Sales Amount (AED)"] or 0) for r in results)
    rows = []
    for r in results:
        kp = r["kpis"]
        tx = int(kp["Transactions"])
        amt = float(kp["Total Sales Amount (AED)"] or 0)
        rows.append(
            {
                "Scenario": r["name"],
                "Transactions": tx,
                "Transaction%": 100.0 * tx / total_tx if total_tx else 0.0,
                "Sales Velocity (txn/day)": kp["Sales Velocity (txn/day)"],
                "Total Sales Amount (AED)": amt,
                "Sales Amount (AED) %": 100.0 * amt / total_amt if total_amt else 0.0,
                "Avg Selling Price (AED)": kp["Avg Selling Price (AED)"],
                "Median Selling Price (AED)": kp["Median Selling Price (AED)"],
                "Avg Size (Sqf)": kp["Avg Size (Sqf)"],
                "Median Size (Sqf)": kp["Median Size (Sqf)"],
                "Mode Size (Sqf)": kp["Mode Size (Sqf)"],
                "Avg PSF (AED/Sqf)": kp["Avg PSF (AED/Sqf)"],
                "Median PSF (AED/Sqf)": kp["Median PSF (AED/Sqf)"],
                "Mode PSF (AED/Sqf)": kp["Mode PSF (AED/Sqf)"],
            }
        )
    return pd.DataFrame(rows)


def top_group_labels(df: pd.DataFrame, group_col: str, n: int, by: str) -> List[str]:
    if df.empty:
        return []
    if by == "amount":
        g = (
            df.groupby(group_col, dropna=False)[C.COL_AMOUNT_AED]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
    else:
        g = df[group_col].value_counts(dropna=False).head(n)
    return [str(x) for x in g.index.tolist()]


def group_kpi_table(df: pd.DataFrame, group_col: str, labels: Sequence[str], scenario_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Rank", group_col, "Scenario"] + KPI_COLUMNS)
    total_tx = int(len(df))
    total_amt = float(df[C.COL_AMOUNT_AED].sum())
    rows = []
    for i, label in enumerate(labels, start=1):
        part = df[df[group_col].astype(str) == str(label)]
        kp = compute_kpis(part)
        tx = int(kp["Transactions"])
        amt = float(kp["Total Sales Amount (AED)"] or 0)
        rows.append(
            {
                "Rank": i,
                group_col: label,
                "Scenario": scenario_name,
                "Transactions": tx,
                "Transaction%": 100.0 * tx / total_tx if total_tx else 0.0,
                "Sales Velocity (txn/day)": kp["Sales Velocity (txn/day)"],
                "Total Sales Amount (AED)": amt,
                "Sales Amount (AED) %": 100.0 * amt / total_amt if total_amt else 0.0,
                "Avg Selling Price (AED)": kp["Avg Selling Price (AED)"],
                "Median Selling Price (AED)": kp["Median Selling Price (AED)"],
                "Avg Size (Sqf)": kp["Avg Size (Sqf)"],
                "Median Size (Sqf)": kp["Median Size (Sqf)"],
                "Mode Size (Sqf)": kp["Mode Size (Sqf)"],
                "Avg PSF (AED/Sqf)": kp["Avg PSF (AED/Sqf)"],
                "Median PSF (AED/Sqf)": kp["Median PSF (AED/Sqf)"],
                "Mode PSF (AED/Sqf)": kp["Mode PSF (AED/Sqf)"],
            }
        )
    return pd.DataFrame(rows)


def overlay_chart(results: Sequence[Dict], series_key: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    for r in results:
        dd = r[series_key]
        fig.add_trace(
            go.Scatter(
                x=dd["Date"],
                y=dd["Value"],
                mode="lines+markers",
                name=r["name"],
            )
        )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time",
        yaxis_title=y_title,
        legend_title="Scenario",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def overlay_bar_chart(results: Sequence[Dict], series_key: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    for r in results:
        dd = r[series_key]
        fig.add_trace(
            go.Bar(
                x=dd["Date"],
                y=dd["Value"],
                name=r["name"],
                opacity=0.65,
            )
        )
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title=title,
        xaxis_title="Time",
        yaxis_title=y_title,
        legend_title="Scenario",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def weighted_avg_ticket(df: pd.DataFrame) -> float:
    tx = float(len(df))
    return float(df[C.COL_AMOUNT_AED].sum()) / tx if tx > 0 else 0.0


def weighted_avg_rate(df: pd.DataFrame) -> float:
    total_size = float(df[C.COL_SIZE_SQF].sum())
    return float(df[C.COL_AMOUNT_AED].sum()) / total_size if total_size > 0 else 0.0


def weighted_avg_size(df: pd.DataFrame) -> float:
    tx = float(len(df))
    return float(df[C.COL_SIZE_SQF].sum()) / tx if tx > 0 else 0.0


def pricing_trend(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Period", "Avg Rate (AED/sqft)", "Avg Ticket (AED)"])
    d = df.copy()
    d["_date"] = pd.to_datetime(d[C.COL_DATE]).dt.normalize()
    if granularity == "Day":
        d["Period"] = d["_date"]
    elif granularity == "Week":
        d["Period"] = d["_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    elif granularity == "Month":
        d["Period"] = d["_date"].dt.to_period("M").dt.start_time
    elif granularity == "Quarter":
        d["Period"] = d["_date"].dt.to_period("Q").dt.start_time
    else:
        d["Period"] = d["_date"].dt.to_period("Y").dt.start_time

    g = d.groupby("Period", as_index=False).agg(
        TotalValue=(C.COL_AMOUNT_AED, "sum"),
        TotalSize=(C.COL_SIZE_SQF, "sum"),
        Transactions=(C.COL_AMOUNT_AED, "size"),
    )
    g["Avg Rate (AED/sqft)"] = np.where(g["TotalSize"] > 0, g["TotalValue"] / g["TotalSize"], 0.0)
    g["Avg Ticket (AED)"] = np.where(g["Transactions"] > 0, g["TotalValue"] / g["Transactions"], 0.0)
    return g.sort_values("Period").reset_index(drop=True)


def top_communities_value(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[C.COL_COMMUNITY, "Total Value (AED)"])
    return (
        df.groupby(C.COL_COMMUNITY, as_index=False)[C.COL_AMOUNT_AED]
        .sum()
        .rename(columns={C.COL_AMOUNT_AED: "Total Value (AED)"})
        .sort_values("Total Value (AED)", ascending=False)
        .head(n)
    )


def top_communities_units_rate(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[C.COL_COMMUNITY, "Units Sold", "Avg Rate (AED/sqft)", "Total Value (AED)"])
    g = df.groupby(C.COL_COMMUNITY, as_index=False).agg(
        **{
            "Units Sold": (C.COL_AMOUNT_AED, "size"),
            "Total Value (AED)": (C.COL_AMOUNT_AED, "sum"),
            "Total Size (Sqf)": (C.COL_SIZE_SQF, "sum"),
        }
    )
    g["Avg Rate (AED/sqft)"] = np.where(g["Total Size (Sqf)"] > 0, g["Total Value (AED)"] / g["Total Size (Sqf)"], 0.0)
    return g.sort_values("Units Sold", ascending=False).head(n)


def top_developers_metrics(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Developer Group", "Value (AED)", "Units", "Avg Rate (AED/sqft)", "Market Share %"])
    total_market_value = float(df[C.COL_AMOUNT_AED].sum())
    g = df.groupby("Developer Group", as_index=False).agg(
        **{
            "Value (AED)": (C.COL_AMOUNT_AED, "sum"),
            "Units": (C.COL_AMOUNT_AED, "size"),
            "Total Size (Sqf)": (C.COL_SIZE_SQF, "sum"),
        }
    )
    g["Avg Rate (AED/sqft)"] = np.where(g["Total Size (Sqf)"] > 0, g["Value (AED)"] / g["Total Size (Sqf)"], 0.0)
    g["Market Share %"] = np.where(total_market_value > 0, 100.0 * g["Value (AED)"] / total_market_value, 0.0)
    return g.sort_values("Value (AED)", ascending=False).head(n)


def bucket_distribution(values: pd.Series, bins: List[float], labels: List[str]) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame({"Bucket": labels, "Count": [0] * len(labels), "Distribution %": [0.0] * len(labels)})
    cat = pd.cut(values.astype(float), bins=bins, labels=labels, include_lowest=True, right=False)
    counts = cat.value_counts(sort=False).reindex(labels, fill_value=0)
    total = int(counts.sum())
    out = pd.DataFrame({"Bucket": counts.index.astype(str), "Count": counts.values})
    out["Distribution %"] = np.where(total > 0, 100.0 * out["Count"] / total, 0.0)
    return out


def render_advanced_analytics(df: pd.DataFrame, scenario_name: str, widget_prefix: str, granularity: str) -> None:
    st.markdown(f"#### {scenario_name}")
    chart_key = st.selectbox(
        "Select advanced view",
        options=[
            "Pricing Trend — Avg Rate & Avg Ticket",
            "Top 10 Communities — Total Sales Value",
            "Top 10 Communities — Units Sold vs Avg Rate",
            "Top Developers — Value, Units & Avg Rate",
            "Price Segmentation — Ticket Distribution",
            "AED/Sqft Distribution",
            "Unit Size Distribution",
        ],
        key=f"adv_chart_{widget_prefix}",
    )

    if chart_key == "Pricing Trend — Avg Rate & Avg Ticket":
        m = pricing_trend(df, granularity=granularity)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=m["Period"],
                y=m["Avg Rate (AED/sqft)"],
                mode="lines+markers",
                name="Average PSF Rate (AED/sqft)",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=m["Period"],
                y=m["Avg Ticket (AED)"],
                mode="lines+markers",
                name="Average Ticket Size (AED)",
                yaxis="y2",
                line=dict(color="#ff7f0e", width=3, dash="dash"),
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=f"Average PSF Rate vs Average Ticket Size by {granularity}",
            xaxis_title=granularity,
            yaxis=dict(title="Average PSF Rate (AED/sqft)"),
            yaxis2=dict(title="Average Ticket Size (AED)", overlaying="y", side="right"),
            legend_title="Metric",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            formatted_kpi_table(m[["Period", "Avg Rate (AED/sqft)", "Avg Ticket (AED)"]]),
            use_container_width=True,
            hide_index=True,
        )

    elif chart_key == "Top 10 Communities — Total Sales Value":
        t = top_communities_value(df, n=10)
        fig = go.Figure(
            go.Bar(
                x=t["Total Value (AED)"],
                y=t[C.COL_COMMUNITY],
                orientation="h",
            )
        )
        fig.update_layout(template="plotly_white", title="Top 10 Communities by Total Sales Value", xaxis_title="Total Value (AED)", yaxis_title="Community", height=450)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        if len(t) >= 2 and float(t.iloc[1]["Total Value (AED)"]) > 0:
            diff = ((float(t.iloc[0]["Total Value (AED)"]) - float(t.iloc[1]["Total Value (AED)"])) / float(t.iloc[1]["Total Value (AED)"])) * 100.0
            st.metric("% Difference (#1 vs #2)", fmt_pct(diff))
        st.dataframe(formatted_kpi_table(t), use_container_width=True, hide_index=True)

    elif chart_key == "Top 10 Communities — Units Sold vs Avg Rate":
        t = top_communities_units_rate(df, n=10)
        size_values = t["Total Value (AED)"] / max(float(t["Total Value (AED)"].max()), 1.0) * 60.0 + 10.0
        fig = go.Figure(
            go.Scatter(
                x=t["Units Sold"],
                y=t["Avg Rate (AED/sqft)"],
                text=t[C.COL_COMMUNITY],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=size_values, sizemode="diameter", opacity=0.7),
            )
        )
        fig.update_layout(template="plotly_white", title="Top 10 Communities: Units vs Avg Rate (Bubble Size = Total Value)", xaxis_title="Units Sold", yaxis_title="Avg Rate (AED/sqft)", height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(formatted_kpi_table(t[[C.COL_COMMUNITY, "Units Sold", "Avg Rate (AED/sqft)", "Total Value (AED)"]]), use_container_width=True, hide_index=True)

    elif chart_key == "Top Developers — Value, Units & Avg Rate":
        t = top_developers_metrics(df, n=10)
        fig = go.Figure(go.Bar(x=t["Developer Group"], y=t["Value (AED)"], name="Value (AED)"))
        fig.update_layout(template="plotly_white", title="Top 10 Developers by Sales Value", xaxis_title="Developer", yaxis_title="Value (AED)", height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(formatted_kpi_table(t[["Developer Group", "Value (AED)", "Units", "Avg Rate (AED/sqft)", "Market Share %"]]), use_container_width=True, hide_index=True)

    elif chart_key == "Price Segmentation — Ticket Distribution":
        bins = [0, 500_000, 700_000, 900_000, 1_200_000, float("inf")]
        labels = ["<500K", "500K-700K", "700K-900K", "900K-1.2M", "1.2M+"]
        t = bucket_distribution(df[C.COL_AMOUNT_AED], bins=bins, labels=labels)
        fig = go.Figure(go.Bar(x=t["Bucket"], y=t["Count"]))
        fig.update_layout(template="plotly_white", title="Ticket Size Distribution", xaxis_title="Ticket Bucket", yaxis_title="Transactions", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(formatted_kpi_table(t), use_container_width=True, hide_index=True)

    elif chart_key == "AED/Sqft Distribution":
        bins = [0, 1500, 2000, 2500, float("inf")]
        labels = ["<1,500", "1,500-2,000", "2,000-2,500", "2,500+"]
        t = bucket_distribution(df[C.COL_AED_PSF], bins=bins, labels=labels)
        fig = go.Figure(go.Bar(x=t["Bucket"], y=t["Count"]))
        fig.update_layout(template="plotly_white", title="AED/Sqft Distribution", xaxis_title="Rate Bucket", yaxis_title="Transactions", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(formatted_kpi_table(t), use_container_width=True, hide_index=True)

    elif chart_key == "Unit Size Distribution":
        bins = [0, 350, 400, 450, 500, float("inf")]
        labels = ["<350", "350-400", "400-450", "450-500", "500+"]
        t = bucket_distribution(df[C.COL_SIZE_SQF], bins=bins, labels=labels)
        fig = go.Figure(go.Bar(x=t["Bucket"], y=t["Count"]))
        fig.update_layout(template="plotly_white", title="Unit Size Distribution", xaxis_title="Size Bucket (Sqf)", yaxis_title="Transactions", height=420)
        st.plotly_chart(fig, use_container_width=True)
        sweet = t.loc[t["Bucket"] == "400-450", "Count"].sum()
        total = t["Count"].sum()
        sweet_pct = (100.0 * sweet / total) if total else 0.0
        st.metric("Sweet Spot % (400-450 sqft)", fmt_pct(sweet_pct))
        st.dataframe(formatted_kpi_table(t), use_container_width=True, hide_index=True)


def psf_bucket_series(psf: pd.Series) -> pd.Series:
    bins = [0, 1500, 2000, 2500, 3000, float("inf")]
    labels = PSF_BUCKET_ORDER
    return pd.cut(psf.astype(float), bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def ticket_bucket_series(amount: pd.Series) -> pd.Series:
    bins = [
        0,
        500_000,
        700_000,
        900_000,
        1_200_000,
        1_500_000,
        2_000_000,
        2_500_000,
        3_000_000,
        4_000_000,
        5_000_000,
        7_000_000,
        10_000_000,
        float("inf"),
    ]
    labels = TICKET_BUCKET_ORDER
    return pd.cut(amount.astype(float), bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def time_bucket_series(date_series: pd.Series, granularity: str) -> pd.Series:
    d = pd.to_datetime(date_series).dt.normalize()
    if granularity == "Day":
        return d.dt.strftime("%Y-%m-%d")
    if granularity == "Week":
        return d.dt.to_period("W-SUN").apply(lambda p: p.start_time.strftime("%Y-%m-%d"))
    if granularity == "Month":
        return d.dt.to_period("M").astype(str)
    if granularity == "Quarter":
        return d.dt.to_period("Q").astype(str)
    return d.dt.to_period("Y").astype(str)


def _apply_axis_order(p: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    out = p.copy()

    def order_for(col_name: str, labels: List[str]) -> List[str]:
        if col_name == "PSF Range":
            ordered = [x for x in PSF_BUCKET_ORDER if x in labels]
            return ordered + [x for x in labels if x not in ordered]
        if col_name == "Ticket Size Range":
            ordered = [x for x in TICKET_BUCKET_ORDER if x in labels]
            return ordered + [x for x in labels if x not in ordered]
        if col_name == C.COL_BEDROOMS:
            ordered = [x for x in C.ALLOWED_BEDROOMS if x in labels]
            return ordered + [x for x in labels if x not in ordered]
        return sorted(labels)

    out = out.reindex(index=order_for(row_col, [str(x) for x in out.index]))
    out = out.reindex(columns=order_for(col_col, [str(x) for x in out.columns]))
    return out


def heatmap_matrix(df: pd.DataFrame, row_col: str, col_col: str, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    # Constrain to top 10 developers by transaction count for all requested heatmaps.
    if "Developer Group" in d.columns:
        top_devs = d["Developer Group"].value_counts().head(10).index.tolist()
        d = d[d["Developer Group"].isin(top_devs)]

    # Constrain community dimension to top 10 communities when community is one axis.
    if row_col == C.COL_COMMUNITY or col_col == C.COL_COMMUNITY:
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        d = d[d[C.COL_COMMUNITY].isin(top_comms)]

    if d.empty:
        return pd.DataFrame()

    if metric == "count":
        p = pd.pivot_table(
            d,
            index=row_col,
            columns=col_col,
            values=C.COL_AMOUNT_AED,
            aggfunc="size",
            fill_value=0,
        )
    elif metric == "avg_psf":
        amt = pd.pivot_table(
            d,
            index=row_col,
            columns=col_col,
            values=C.COL_AMOUNT_AED,
            aggfunc="sum",
            fill_value=0.0,
        )
        size = pd.pivot_table(
            d,
            index=row_col,
            columns=col_col,
            values=C.COL_SIZE_SQF,
            aggfunc="sum",
            fill_value=0.0,
        )
        p = amt.div(size.replace(0, np.nan)).fillna(0.0)
    else:
        p = pd.pivot_table(
            d,
            index=row_col,
            columns=col_col,
            values=C.COL_AMOUNT_AED,
            aggfunc="sum",
            fill_value=0.0,
        )
    return _apply_axis_order(p, row_col=row_col, col_col=col_col)


def render_heatmap(df: pd.DataFrame, row_col: str, col_col: str, metric: str, title: str) -> None:
    p = heatmap_matrix(df=df, row_col=row_col, col_col=col_col, metric=metric)
    if p.empty:
        st.info("No filtered data available for this heatmap.")
        return
    if metric == "count":
        colorbar_title = "Transactions"
    elif metric == "avg_psf":
        colorbar_title = "Avg PSF (AED/sqft)"
    else:
        colorbar_title = "Total Value (AED)"
    fig = go.Figure(
        data=go.Heatmap(
            z=p.values,
            x=p.columns.astype(str),
            y=p.index.astype(str),
            colorscale="Blues",
            hoverongaps=False,
            colorbar=dict(title=colorbar_title),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis=dict(title=col_col, categoryorder="array", categoryarray=[str(x) for x in p.columns]),
        yaxis=dict(title=row_col, categoryorder="array", categoryarray=[str(x) for x in p.index]),
        height=520,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_heatmaps_for_df(df: pd.DataFrame, section_key: str, granularity: str) -> None:
    d = df.copy()
    d["PSF Range"] = psf_bucket_series(d[C.COL_AED_PSF]) if not d.empty else pd.Series(dtype=str)
    d["Ticket Size Range"] = ticket_bucket_series(d[C.COL_AMOUNT_AED]) if not d.empty else pd.Series(dtype=str)
    d["Time Bucket"] = time_bucket_series(d[C.COL_DATE], granularity=granularity) if not d.empty else pd.Series(dtype=str)

    choice = st.selectbox(
        "Select heatmap",
        options=[
            "Developer vs Community — Transactions",
            "Developer vs Community — Total Value",
            "Community vs Developer — Average PSF Rate",
            "Developer vs Time — Transactions",
            "Developer vs Time — Total Value",
            "Developer vs Unit Type — Total Value",
            "Developer vs PSF Range — Total Value",
            "Community vs Unit Type — Total Value",
            "Community vs PSF Range — Total Value",
            "Developer vs Ticket Size — Total Value",
            "Community vs Ticket Size — Total Value",
            "Ticket Size vs Time — Transactions",
            "Community vs Time — Transactions",
            "Community vs Time — Total Value",
            "Community vs Time — Average PSF Rate",
        ],
        key=f"heatmap_choice_{section_key}",
    )

    if choice == "Developer vs Community — Transactions":
        render_heatmap(
            d,
            row_col="Developer Group",
            col_col=C.COL_COMMUNITY,
            metric="count",
            title="Developer vs Community Heatmap (No. of Transactions)",
        )
    elif choice == "Developer vs Community — Total Value":
        render_heatmap(
            d,
            row_col="Developer Group",
            col_col=C.COL_COMMUNITY,
            metric="sum",
            title="Developer vs Community Heatmap (Total Value of Transactions)",
        )
    elif choice == "Community vs Developer — Average PSF Rate":
        render_heatmap(
            d,
            row_col=C.COL_COMMUNITY,
            col_col="Developer Group",
            metric="avg_psf",
            title="Community vs Developer Heatmap (Average PSF Rate)",
        )
    elif choice == "Developer vs Unit Type — Total Value":
        render_heatmap(
            d,
            row_col="Developer Group",
            col_col=C.COL_BEDROOMS,
            metric="sum",
            title="Developer vs Unit Type Heatmap (Total Value of Transactions)",
        )
    elif choice == "Developer vs PSF Range — Total Value":
        render_heatmap(
            d,
            row_col="Developer Group",
            col_col="PSF Range",
            metric="sum",
            title="Developer vs PSF Range Heatmap (Total Value of Transactions)",
        )
    elif choice == "Developer vs Time — Transactions":
        top_devs = d["Developer Group"].value_counts().head(10).index.tolist()
        dd = d[d["Developer Group"].isin(top_devs)].copy()
        render_heatmap(
            dd,
            row_col="Developer Group",
            col_col="Time Bucket",
            metric="count",
            title=f"Top 10 Developers vs {granularity} Heatmap (No. of Transactions)",
        )
    elif choice == "Developer vs Time — Total Value":
        top_devs = d["Developer Group"].value_counts().head(10).index.tolist()
        dd = d[d["Developer Group"].isin(top_devs)].copy()
        render_heatmap(
            dd,
            row_col="Developer Group",
            col_col="Time Bucket",
            metric="sum",
            title=f"Top 10 Developers vs {granularity} Heatmap (Total Value of Transactions)",
        )
    elif choice == "Community vs Unit Type — Total Value":
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col=C.COL_BEDROOMS,
            metric="sum",
            title="Top 10 Communities vs Unit Type Heatmap (Total Value of Transactions)",
        )
    elif choice == "Community vs PSF Range — Total Value":
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col="PSF Range",
            metric="sum",
            title="Top 10 Communities vs PSF Range Heatmap (Total Value of Transactions)",
        )
    elif choice == "Developer vs Ticket Size — Total Value":
        top_devs = d["Developer Group"].value_counts().head(10).index.tolist()
        dd = d[d["Developer Group"].isin(top_devs)].copy()
        render_heatmap(
            dd,
            row_col="Developer Group",
            col_col="Ticket Size Range",
            metric="sum",
            title="Top 10 Developers vs Average Ticket Size Range Heatmap (Total Value of Transactions)",
        )
    elif choice == "Community vs Ticket Size — Total Value":
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col="Ticket Size Range",
            metric="sum",
            title="Top 10 Communities vs Average Ticket Size Range Heatmap (Total Value of Transactions)",
        )
    elif choice == "Ticket Size vs Time — Transactions":
        render_heatmap(
            d,
            row_col="Ticket Size Range",
            col_col="Time Bucket",
            metric="count",
            title=f"Average Ticket Size Range vs {granularity} Heatmap (No. of Transactions)",
        )
    elif choice == "Community vs Time — Transactions":
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col="Time Bucket",
            metric="count",
            title=f"Top 10 Communities vs {granularity} Heatmap (No. of Transactions)",
        )
    elif choice == "Community vs Time — Total Value":
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col="Time Bucket",
            metric="sum",
            title=f"Top 10 Communities vs {granularity} Heatmap (Total Value of Transactions)",
        )
    else:
        top_comms = d[C.COL_COMMUNITY].value_counts().head(10).index.tolist()
        dd = d[d[C.COL_COMMUNITY].isin(top_comms)].copy()
        render_heatmap(
            dd,
            row_col=C.COL_COMMUNITY,
            col_col="Time Bucket",
            metric="avg_psf",
            title=f"Top 10 Communities vs {granularity} Heatmap (Average PSF Rate)",
        )



st.title("Dubai Real Estate Transactions Dashboard")
st.caption("Executive analytics with cascading filters and multi-scenario comparison.")

with st.sidebar:
    st.subheader("Data")
    rebuild = st.button("Rebuild cleaned dataset now")
    df, audit = load_data(force_rebuild=rebuild)
    latest_date = pd.to_datetime(df[C.COL_DATE]).max().date()
    st.markdown(f"**Latest transaction date:** <span class='sobha-accent'>{latest_date}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='small-note'>Rows (cleaned): {len(df):,}</span>", unsafe_allow_html=True)
    if audit:
        with st.expander("ETL audit log"):
            st.json(audit)

st.sidebar.subheader("Scenarios")
num_scenarios = st.sidebar.slider("Number of scenarios", min_value=1, max_value=5, value=1)
top_n_choice = st.sidebar.selectbox("Hotspots table size", options=[10, 20], index=0)

scenarios: List[Dict] = []
for i in range(num_scenarios):
    with st.expander(f"Scenario {i + 1} filters", expanded=(i == 0)):
        scenario_name, scenario_filters = scenario_filter_widgets(
            prefix=f"s{i + 1}",
            label_default=f"Scenario {i + 1}",
            df=df,
        )
    scenarios.append({"name": scenario_name, "filters": scenario_filters})

date_mins = [pd.to_datetime(s["filters"].date_min) for s in scenarios]
date_maxs = [pd.to_datetime(s["filters"].date_max) for s in scenarios]
range_span_days = int((max(date_maxs) - min(date_mins)).days) + 1
granularity_options = available_granularities(range_span_days)
st.markdown("### Trend Controls")
granularity = st.radio(
    "Granularity",
    options=granularity_options,
    horizontal=True,
    index=0,
)

scenario_results: List[Dict] = []
for s in scenarios:
    dff = apply_filters(df, s["filters"])
    cnt_series = aggregate_series(dff, basis="count", granularity=granularity)
    amt_series = aggregate_series(dff, basis="amount", granularity=granularity)
    scenario_results.append(
        {
            "name": s["name"],
            "df": dff,
            "kpis": compute_kpis(dff),
            "series_cnt": cnt_series,
            "series_amt": amt_series,
            "series_cnt_cum": cumulative_series(cnt_series),
            "series_amt_cum": cumulative_series(amt_series),
        }
    )

if len(scenario_results) > 1:
    dashboard_df = pd.concat([r["df"] for r in scenario_results], ignore_index=True)
    st.caption("KPI cards below are combined across all selected scenarios.")
else:
    dashboard_df = scenario_results[0]["df"]

dashboard_kpis = compute_kpis(dashboard_df)
kpi_cols = st.columns(6)
kpi_cols[0].metric("Transactions", fmt_whole(dashboard_kpis["Transactions"]))
kpi_cols[1].metric("Sales Velocity", f"{fmt_whole(dashboard_kpis['Sales Velocity (txn/day)'])} txn/day")
kpi_cols[2].metric("Total Sales", fmt_aed_compact(dashboard_kpis["Total Sales Amount (AED)"]))
kpi_cols[3].metric("Avg Selling Price", fmt_aed_compact(dashboard_kpis["Avg Selling Price (AED)"]))
kpi_cols[4].metric("Median Selling Price", fmt_aed_compact(dashboard_kpis["Median Selling Price (AED)"]))
kpi_cols[5].metric("Median PSF", fmt_whole(dashboard_kpis["Median PSF (AED/Sqf)"]))

st.markdown("### Scenario KPI Comparison")
cmp_df = scenario_kpi_rows(scenario_results)
st.dataframe(formatted_kpi_table(cmp_df), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Trends (Weekdays only)")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        overlay_bar_chart(
            scenario_results,
            "series_cnt",
            f"Transactions by {granularity} (sum)",
            "Transactions",
        ),
        use_container_width=True,
    )
with c2:
    st.plotly_chart(
        overlay_bar_chart(
            scenario_results,
            "series_amt",
            f"Sales Amount by {granularity} (sum)",
            "Amount (AED)",
        ),
        use_container_width=True,
    )

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        overlay_chart(
            scenario_results,
            "series_cnt_cum",
            f"Cumulative Transactions by {granularity}",
            "Transactions",
        ),
        use_container_width=True,
    )
with c4:
    st.plotly_chart(
        overlay_chart(
            scenario_results,
            "series_amt_cum",
            f"Cumulative Sales Amount by {granularity}",
            "Amount (AED)",
        ),
        use_container_width=True,
    )

st.markdown("---")
st.subheader("Advanced Analytics")
st.caption("Weighted formulas are used: Avg Ticket = Total Value / Transactions, Avg Rate = Total Value / Total Size, Avg Size = Total Size / Transactions.")
advanced_scope = st.radio(
    "Advanced analytics scope",
    options=["Selected scenario", "All scenarios"],
    horizontal=True,
)
if advanced_scope == "Selected scenario":
    selected_name = st.selectbox("Scenario", options=[r["name"] for r in scenario_results], key="adv_selected_scenario")
    selected_result = next(r for r in scenario_results if r["name"] == selected_name)
    selected_idx = next(i for i, r in enumerate(scenario_results) if r["name"] == selected_name)
    render_advanced_analytics(
        selected_result["df"],
        selected_result["name"],
        widget_prefix=f"selected_{selected_idx}",
        granularity=granularity,
    )
else:
    adv_tabs = st.tabs([r["name"] for r in scenario_results])
    for i, tab in enumerate(adv_tabs):
        with tab:
            render_advanced_analytics(
                scenario_results[i]["df"],
                scenario_results[i]["name"],
                widget_prefix=f"all_{i}",
                granularity=granularity,
            )

st.markdown("---")
st.subheader("Heatmaps")
st.caption("Heatmaps use only the filtered rows of the selected scenario scope.")
heatmap_scope = st.radio(
    "Heatmap scope",
    options=["Selected scenario", "All scenarios", "Combined selected scenarios"],
    horizontal=True,
    key="heatmap_scope",
)
if heatmap_scope == "Selected scenario":
    hm_name = st.selectbox("Scenario", options=[r["name"] for r in scenario_results], key="heatmap_selected_scenario")
    hm_result = next(r for r in scenario_results if r["name"] == hm_name)
    hm_idx = next(i for i, r in enumerate(scenario_results) if r["name"] == hm_name)
    render_heatmaps_for_df(hm_result["df"], section_key=f"selected_{hm_idx}", granularity=granularity)
elif heatmap_scope == "All scenarios":
    hm_tabs = st.tabs([r["name"] for r in scenario_results])
    for i, tab in enumerate(hm_tabs):
        with tab:
            render_heatmaps_for_df(scenario_results[i]["df"], section_key=f"all_{i}", granularity=granularity)
else:
    combined_df = pd.concat([r["df"] for r in scenario_results], ignore_index=True) if scenario_results else pd.DataFrame()
    render_heatmaps_for_df(combined_df, section_key="combined", granularity=granularity)

st.markdown("---")
st.subheader("Hotspots by Scenario")
scenario_tabs = st.tabs([r["name"] for r in scenario_results])
for idx, tab in enumerate(scenario_tabs):
    res = scenario_results[idx]
    dff = res["df"]
    with tab:
        st.markdown(f"**Top {top_n_choice} Communities by Transactions**")
        labels = top_group_labels(dff, C.COL_COMMUNITY, top_n_choice, by="count")
        table = group_kpi_table(dff, C.COL_COMMUNITY, labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

        st.markdown(f"**Top {top_n_choice} Communities by Sales Amount**")
        labels = top_group_labels(dff, C.COL_COMMUNITY, top_n_choice, by="amount")
        table = group_kpi_table(dff, C.COL_COMMUNITY, labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

        st.markdown(f"**Top {top_n_choice} Developers by Transactions**")
        labels = top_group_labels(dff, "Developer Group", top_n_choice, by="count")
        table = group_kpi_table(dff, "Developer Group", labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

        st.markdown(f"**Top {top_n_choice} Developers by Sales Amount**")
        labels = top_group_labels(dff, "Developer Group", top_n_choice, by="amount")
        table = group_kpi_table(dff, "Developer Group", labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

        st.markdown(f"**Top {top_n_choice} Properties / Projects by Transactions**")
        labels = top_group_labels(dff, C.COL_PROPERTY, top_n_choice, by="count")
        table = group_kpi_table(dff, C.COL_PROPERTY, labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

        st.markdown(f"**Top {top_n_choice} Properties / Projects by Sales Amount**")
        labels = top_group_labels(dff, C.COL_PROPERTY, top_n_choice, by="amount")
        table = group_kpi_table(dff, C.COL_PROPERTY, labels, scenario_name=res["name"])
        st.dataframe(formatted_kpi_table(table), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Transactions by Scenario")
tx_tabs = st.tabs([r["name"] for r in scenario_results])
display_cols = [
    C.COL_DATE,
    "Transaction Bucket",
    C.COL_PROPERTY_TYPE,
    C.COL_COMMUNITY,
    C.COL_PROPERTY,
    C.COL_BEDROOMS,
    C.COL_SIZE_SQF,
    C.COL_AMOUNT_AED,
    C.COL_AED_PSF,
    "Developer Group",
    "Developer Raw",
    C.COL_TIMES_SOLD,
]
for i, tab in enumerate(tx_tabs):
    with tab:
        dff = scenario_results[i]["df"]
        summary_row = scenario_kpi_rows([scenario_results[i]])
        st.markdown("**Scenario KPI Summary**")
        st.dataframe(formatted_kpi_table(summary_row), use_container_width=True, hide_index=True)

        st.markdown("**Transaction Details**")
        available_cols = [c for c in display_cols if c in dff.columns]
        st.dataframe(dff[available_cols].head(2000), use_container_width=True, hide_index=True)

st.markdown(
    "<div class='small-note'>Tip: categorical filters are cascading in order and only show values valid under prior selections.</div>",
    unsafe_allow_html=True,
)
