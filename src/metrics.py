
"""
Metrics + helper functions used by the dashboard.

Everything here is "pure python" (no Streamlit) so it's easy to test/reuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Filters:
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None

    transaction_bucket: str = "Combined"  # Combined | Ready Property | Off-Plan Property
    property_type: str = "Combined"       # Combined | Villa | Apartment

    bedrooms: Optional[List[str]] = None
    community: Optional[List[str]] = None
    developer_group: Optional[List[str]] = None
    prop: Optional[List[str]] = None

    size_range: Optional[Tuple[float, float]] = None
    amount_range: Optional[Tuple[float, float]] = None
    psf_range: Optional[Tuple[float, float]] = None

    include_outliers: bool = False


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()

    # Dates
    if f.date_min is not None:
        out = out[pd.to_datetime(out[C.COL_DATE]) >= pd.to_datetime(f.date_min)]
    if f.date_max is not None:
        out = out[pd.to_datetime(out[C.COL_DATE]) <= pd.to_datetime(f.date_max)]

    # Transaction bucket
    if f.transaction_bucket and f.transaction_bucket != "Combined":
        out = out[out["Transaction Bucket"] == f.transaction_bucket]

    # Property type
    if f.property_type and f.property_type != "Combined":
        out = out[out[C.COL_PROPERTY_TYPE] == f.property_type]

    # Bedrooms
    if f.bedrooms:
        out = out[out[C.COL_BEDROOMS].isin(f.bedrooms)]

    # Community / Developer / Property
    if f.community:
        out = out[out[C.COL_COMMUNITY].isin(f.community)]
    if f.developer_group:
        out = out[out["Developer Group"].isin(f.developer_group)]
    if f.prop:
        out = out[out[C.COL_PROPERTY].isin(f.prop)]

    # Ranges
    if f.size_range:
        lo, hi = f.size_range
        out = out[(out[C.COL_SIZE_SQF] >= lo) & (out[C.COL_SIZE_SQF] <= hi)]
    if f.amount_range:
        lo, hi = f.amount_range
        out = out[(out[C.COL_AMOUNT_AED] >= lo) & (out[C.COL_AMOUNT_AED] <= hi)]
    if f.psf_range:
        lo, hi = f.psf_range
        out = out[(out[C.COL_AED_PSF] >= lo) & (out[C.COL_AED_PSF] <= hi)]

    # Outliers
    if not f.include_outliers and "Any Outlier" in out.columns:
        out = out[~out["Any Outlier"]]

    return out


def safe_mode(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    modes = s.mode()
    if modes.empty:
        return None
    return float(modes.iloc[0])


def compute_kpis(df: pd.DataFrame) -> Dict[str, float | int | None]:
    """
    KPI definitions match the user's requested metrics.
    """
    if df.empty:
        return {
            "Transactions": 0,
            "Sales Velocity (txn/day)": 0.0,
            "Total Sales Amount (AED)": 0.0,
            "Avg Selling Price (AED)": None,
            "Median Selling Price (AED)": None,
            "Mode Selling Price (AED)": None,
            "Avg Size (Sqf)": None,
            "Median Size (Sqf)": None,
            "Mode Size (Sqf)": None,
            "Avg PSF (AED/Sqf)": None,
            "Median PSF (AED/Sqf)": None,
            "Mode PSF (AED/Sqf)": None,
        }

    tx_count = int(len(df))
    total_amount = float(df[C.COL_AMOUNT_AED].sum())

    date_min = pd.to_datetime(df[C.COL_DATE]).min().normalize()
    date_max = pd.to_datetime(df[C.COL_DATE]).max().normalize()
    days = int((date_max - date_min).days) + 1
    days = max(days, 1)
    velocity = tx_count / days

    avg_price = float(df[C.COL_AMOUNT_AED].mean())
    med_price = float(df[C.COL_AMOUNT_AED].median())
    mode_price = safe_mode(df[C.COL_AMOUNT_AED])

    avg_size = float(df[C.COL_SIZE_SQF].mean())
    med_size = float(df[C.COL_SIZE_SQF].median())
    mode_size = safe_mode(df[C.COL_SIZE_SQF])

    # Avg PSF requested as Total Amount / Total Size
    total_size = float(df[C.COL_SIZE_SQF].sum())
    avg_psf = (total_amount / total_size) if total_size > 0 else None
    med_psf = float(df[C.COL_AED_PSF].median())
    mode_psf = safe_mode(df[C.COL_AED_PSF])

    return {
        "Transactions": tx_count,
        "Sales Velocity (txn/day)": float(velocity),
        "Total Sales Amount (AED)": total_amount,
        "Avg Selling Price (AED)": avg_price,
        "Median Selling Price (AED)": med_price,
        "Mode Selling Price (AED)": mode_price,
        "Avg Size (Sqf)": avg_size,
        "Median Size (Sqf)": med_size,
        "Mode Size (Sqf)": mode_size,
        "Avg PSF (AED/Sqf)": avg_psf,
        "Median PSF (AED/Sqf)": med_psf,
        "Mode PSF (AED/Sqf)": mode_psf,
    }


def daily_series(df: pd.DataFrame, basis: str = "count") -> pd.DataFrame:
    """
    Returns a complete daily timeseries with missing dates filled as 0.

    basis:
      - "count": daily number of transactions
      - "amount": daily sum of Amount (AED)
    """
    if df.empty:
        return pd.DataFrame({"Date": [], "Value": []})

    d = df.copy()
    d["Date"] = pd.to_datetime(d[C.COL_DATE]).dt.normalize()

    if basis == "amount":
        grp = d.groupby("Date", as_index=False)[C.COL_AMOUNT_AED].sum().rename(columns={C.COL_AMOUNT_AED: "Value"})
    else:
        grp = d.groupby("Date", as_index=False).size().rename(columns={"size": "Value"})

    # fill missing dates
    date_min = grp["Date"].min()
    date_max = grp["Date"].max()
    all_days = pd.date_range(date_min, date_max, freq="D")
    grp = grp.set_index("Date").reindex(all_days, fill_value=0).rename_axis("Date").reset_index()
    return grp


def dry_period_stats(
    daily_df: pd.DataFrame,
    x_percent: float,
    peak_quantile: float = 1.0,
) -> Dict[str, float | int]:
    """
    Dry period = consecutive days where daily sales are <= (x% of reference peak).

    reference peak:
      - If peak_quantile == 1.0: max(daily sales)
      - Else: quantile(peak_quantile) of daily sales (e.g. 0.8 for "80th percentile peak")

    Returns:
      - threshold_value
      - longest_dry_streak_days
      - dry_days_total
      - current_dry_streak_days (streak ending on the last day in the series)
    """
    if daily_df.empty:
        return {
            "threshold_value": 0.0,
            "longest_dry_streak_days": 0,
            "dry_days_total": 0,
            "current_dry_streak_days": 0,
        }

    vals = daily_df["Value"].astype(float)
    if peak_quantile >= 1.0:
        ref_peak = float(vals.max())
    else:
        ref_peak = float(vals.quantile(peak_quantile))

    threshold = (x_percent / 100.0) * ref_peak

    is_dry = vals <= threshold

    longest = 0
    current = 0
    for dry in is_dry.tolist():
        if dry:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    # current streak at end
    current_end = 0
    for dry in reversed(is_dry.tolist()):
        if dry:
            current_end += 1
        else:
            break

    return {
        "threshold_value": float(threshold),
        "longest_dry_streak_days": int(longest),
        "dry_days_total": int(is_dry.sum()),
        "current_dry_streak_days": int(current_end),
    }


def top_n(
    df: pd.DataFrame,
    group_col: str,
    n: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      - by_count: top n by transactions count
      - by_amount: top n by total Amount(AED)
    """
    if df.empty:
        empty = pd.DataFrame(columns=[group_col, "Transactions", "Total Amount (AED)"])
        return {"by_count": empty, "by_amount": empty}

    g = df.groupby(group_col, dropna=False).agg(
        Transactions=(C.COL_AMOUNT_AED, "size"),
        **{"Total Amount (AED)": (C.COL_AMOUNT_AED, "sum")}
    ).reset_index()

    by_count = g.sort_values(["Transactions", "Total Amount (AED)"], ascending=[False, False]).head(n)
    by_amount = g.sort_values(["Total Amount (AED)", "Transactions"], ascending=[False, False]).head(n)
    return {"by_count": by_count, "by_amount": by_amount}

