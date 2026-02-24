
"""
Utilities for normalizing and fuzzy-grouping developer names.

Goal:
- "Damac Properties", "Damac Maison"  -> "Damac"
- Reduce legal suffix noise (PJSC, LLC, L.L.C, etc.)
- Keep a transparent path for manual overrides (config/developer_aliases.csv)
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

LEGAL_NOISE_PATTERNS = [
    r"\b(pjsc|llc|l\.l\.c|ltd|limited|inc|co|company)\b",
    r"\b(properties?|property)\b",
    r"\b(development|developers?)\b",
    r"\b(real\s*estate)\b",
    r"\b(investment|investments)\b",
    r"\b(holding|holdings)\b",
    r"\b(group)\b",
    r"\b(p\.?j\.?s\.?c\.?)\b",
]

PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]+")

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = PUNCTUATION_RE.sub(" ", s)
    for pat in LEGAL_NOISE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def load_alias_map(csv_path: str) -> Dict[str, str]:
    """
    Optional manual override file (2 columns):
    alias,canonical

    Example:
    Damac Maison,Damac
    Damac Properties,Damac
    """
    import pandas as pd

    try:
        ali = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {}

    ali = ali.dropna(subset=["alias", "canonical"])
    return {str(a).strip(): str(c).strip() for a, c in zip(ali["alias"], ali["canonical"])}


def best_fuzzy_match(
    raw_name: str,
    canonical_names: Iterable[str],
    threshold: int = 90,
) -> Tuple[Optional[str], int]:
    """
    Returns (best_match, score). If score < threshold -> (None, score)

    Uses RapidFuzz token_set_ratio if available, else falls back to a crude
    overlap scoring.
    """
    raw_norm = _normalize_text(raw_name)
    if not raw_norm:
        return None, 0

    best_name = None
    best_score = 0

    for canon in canonical_names:
        canon_norm = _normalize_text(canon)
        if not canon_norm:
            continue

        if fuzz is not None:
            score = int(fuzz.token_set_ratio(raw_norm, canon_norm))
        else:  # fallback
            raw_tokens = set(raw_norm.split())
            canon_tokens = set(canon_norm.split())
            inter = len(raw_tokens.intersection(canon_tokens))
            union = max(1, len(raw_tokens.union(canon_tokens)))
            score = int(100 * inter / union)

        # extra boost if one contains the other
        if raw_norm in canon_norm or canon_norm in raw_norm:
            score = max(score, 95)

        if score > best_score:
            best_score = score
            best_name = canon

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def normalize_developer(
    raw_name: str,
    canonical_names: Iterable[str],
    alias_map: Optional[Dict[str, str]] = None,
    threshold: int = 90,
) -> str:
    """
    Returns a clean, grouped developer label.

    Order of precedence:
    1) manual alias map exact match (case-insensitive)
    2) fuzzy match to canonical list
    3) cleaned original (title-cased) as fallback
    """
    if raw_name is None:
        return "Unknown"

    raw_str = str(raw_name).strip()
    if not raw_str:
        return "Unknown"

    # Manual override
    if alias_map:
        # case-insensitive exact match
        for k, v in alias_map.items():
            if raw_str.lower() == str(k).strip().lower():
                return v

    # Fuzzy match to canonical
    match, _score = best_fuzzy_match(raw_str, canonical_names, threshold=threshold)
    if match:
        return match

    # Fallback: clean and title-case the de-noised name (but keep some structure)
    cleaned = _normalize_text(raw_str)
    if not cleaned:
        return "Unknown"
    # Title case but keep acronyms-ish
    return " ".join([t.upper() if len(t) <= 3 else t.capitalize() for t in cleaned.split()])

