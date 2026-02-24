
"""
Project configuration for the Dubai real estate transactions dashboard.

You can safely tune thresholds and allowed values here without touching the app code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ---- Columns (as they appear in the provided Excel export) ----
COL_TRANSACTION_TYPE = "Transaction Type"
COL_DATE = "Date"
COL_PROPERTY_TYPE = "Property Type"
COL_BEDROOMS = "Bedrooms"
COL_DEVELOPER = "Developer"
COL_COMMUNITY = "Community"
COL_PROPERTY = "Property"
COL_TIMES_SOLD = "Times Sold"
COL_SIZE_SQF = "Size (Sqf)"
COL_AMOUNT_AED = "Amount (AED)"
COL_AED_PSF = "AED/Sqf"

# ---- Allowed values / filtering logic ----
ALLOWED_PROPERTY_TYPES: List[str] = ["Villa", "Apartment"]

ALLOWED_BEDROOMS: List[str] = [
    "0 B/R",
    "1 B/R",
    "2 B/R",
    "3 B/R",
    "4 B/R",
    "5 B/R",
    "6 B/R",
    "7 B/R",
    "8 B/R",
    "9 B/R",
    "10 B/R",
    "PENTHOUSE",
]

# Transaction buckets (Ready vs Off-plan)
TX_BUCKET_READY = "Ready Property"
TX_BUCKET_OFFPLAN = "Off-Plan Property"
TX_BUCKET_OTHER = "Other / Unknown"

# For the "compare" dropdowns
TX_BUCKET_FILTER_OPTIONS = ["Combined", TX_BUCKET_READY, TX_BUCKET_OFFPLAN]
PROPERTY_TYPE_FILTER_OPTIONS = ["Combined"] + ALLOWED_PROPERTY_TYPES


# ---- Developer normalization ----
# These are the *canonical* names we want to report by (you can add more).
# Anything fuzzy-matched to one of these will be mapped to that canonical label.
CANONICAL_DEVELOPERS: List[str] = [
    "Emaar",
    "Damac",
    "Binghatti",
    "Sobha",
    "Nakheel",
    "Meraas",
    "Dubai Properties",
    "Azizi",
    "Danube",
    "Samana",
    "Ellington",
    "Imtiaz",
    "Aldar",
    "Omniyat",
    "Arada",
    "H&H",
    "Nshama",
    "Select Group",
    "Deyaar",
    "Tiger",
    "Majid Al Futtaim",
]

# Similarity threshold for fuzzy matching (0-100).
# Higher = stricter matching.
DEVELOPER_FUZZY_THRESHOLD: int = 90


# ---- Outlier handling ----
@dataclass(frozen=True)
class OutlierConfig:
    enabled: bool = True
    # IQR method: keep rows within [Q1 - k_low*IQR, Q3 + k_high*IQR]
    k_low: float = 1.5
    k_high: float = 3.0


DEFAULT_OUTLIER_CONFIG = OutlierConfig()


# ---- Dry-period logic ----
@dataclass(frozen=True)
class DryPeriodConfig:
    # Dry threshold is: (x_percent / 100) * reference_peak
    # reference_peak is max(daily_sales) by default. If you want a more robust
    # definition of "peak", set peak_quantile=0.8 (80th percentile).
    peak_quantile: float = 1.0
    # Whether we calculate dry days based on transaction COUNT or on total AMOUNT (AED).
    basis: str = "count"  # "count" or "amount"


DEFAULT_DRY_CONFIG = DryPeriodConfig()


# ---- Input parsing defaults ----
DEFAULT_SHEET_NAME = "Export"

