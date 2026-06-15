"""utils.py — Shared utilities used by feature/score/reasoning modules.

ponytail: keep this thin. If a function is only used by one module, prefer
inlining it there. We accept the duplication of two-line helpers between
modules in exchange for fewer cross-file dependencies.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence


# ----------------------------------------------------------------------------
# String helpers
# ----------------------------------------------------------------------------

def lc(s) -> str:
    """Lowercase, treat None as empty string."""
    return (s or "").lower()


def in_any(text: str, needles: Sequence[str]) -> bool:
    """True if `text` contains any of `needles` (case-insensitive substring)."""
    if not text:
        return False
    text = text.lower()
    return any(n in text for n in needles)


def count_any(text: str, needles: Sequence[str]) -> int:
    """Count how many of `needles` appear as substrings in `text`."""
    if not text:
        return 0
    text = text.lower()
    return sum(1 for n in needles if n in text)


def first_present(text: str, needles: Sequence[str]) -> str | None:
    """Return the first needle that appears in text (case-insensitive), or None."""
    if not text:
        return None
    t = text.lower()
    for n in needles:
        if n in t:
            return n
    return None


# ----------------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------------

def safe_log1p(x: float) -> float:
    """log1p clamped to non-negative. log1p(-1) would NaN; we floor at 0."""
    return math.log1p(max(0.0, x))


def log_normalize(x: float, ceiling: float) -> float:
    """Compress x to [0, 1] using log1p, normalized against a ceiling.

    A value of `ceiling` maps to 1.0. 0 maps to 0.0. Useful for signals
    with long tails (search_appearance_30d, saved_by_recruiters_30d, etc.)
    """
    if x <= 0:
        return 0.0
    if ceiling <= 0:
        return 0.0
    return min(1.0, safe_log1p(x) / safe_log1p(ceiling))


def inverse_normalize(x: float, ceiling: float) -> float:
    """Inverse signal: high=bad, low=good. Returns [0, 1] with 1=best.

    For avg_response_time_hours: 0 hours → 1.0, ceiling hours → 0.0.
    """
    if x <= 0:
        return 1.0  # missing/instant is treated as good (open to work)
    if ceiling <= 0:
        return 1.0
    return max(0.0, 1.0 - min(1.0, x / ceiling))


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ----------------------------------------------------------------------------
# Lists / aggregations
# ----------------------------------------------------------------------------

def list_field(obj, field: str, default=None) -> list:
    """Read a list field safely, returning default if missing/non-list."""
    v = obj.get(field) if isinstance(obj, dict) else None
    return v if isinstance(v, list) else (default or [])


def str_field(obj, field: str, default: str = "") -> str:
    """Read a string field safely, defaulting to empty string."""
    v = obj.get(field) if isinstance(obj, dict) else None
    return v if isinstance(v, str) else default


def num_field(obj, field: str, default: float = 0.0) -> float:
    """Read a numeric field safely, coercing None to default."""
    v = obj.get(field) if isinstance(obj, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return default


def bool_field(obj, field: str, default: bool = False) -> bool:
    v = obj.get(field) if isinstance(obj, dict) else None
    return v if isinstance(v, bool) else default


def safe_max(values: Iterable[float], default: float = 0.0) -> float:
    """max() of an iterable, returning default if empty."""
    try:
        return max(values)
    except (ValueError, TypeError):
        return default


def sum_field(items: list, field: str) -> float:
    """Sum a numeric field over a list of dicts, ignoring missing/non-numeric."""
    total = 0.0
    for it in items:
        if isinstance(it, dict):
            v = it.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total += v
    return total


# ----------------------------------------------------------------------------
# Truncation (for safe reasoning strings)
# ----------------------------------------------------------------------------

def truncate_reasoning(s: str, max_len: int) -> str:
    """Trim to max_len, ellipsis if cut. Avoid producing >300 char strings."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


# ----------------------------------------------------------------------------
# Build a "summary" string for keyword matching.
# ----------------------------------------------------------------------------

def build_search_blob(candidate: dict) -> str:
    """Build a single lowercased text blob from a candidate for fast matching.

    The blob includes headline, summary, current title, all career
    descriptions, and skill names. We don't need punctuation handling
    for substring matching; the lower() handles case.

    ponytail: building the blob per-candidate is O(text_length). For
    100K candidates this is ~3s of work — acceptable inside the
    5-minute budget. If we needed more speed, we could pre-tokenize.
    """
    parts: List[str] = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("current_company", ""))
    parts.append(profile.get("current_industry", ""))

    for c in candidate.get("career_history", []):
        parts.append(c.get("title", ""))
        parts.append(c.get("company", ""))
        parts.append(c.get("description", ""))

    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))

    return " ".join(parts).lower()
