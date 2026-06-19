"""scoring.py — Composite score from candidate dict.

Per audit spec (2026-06-19):
    final_score = (fit_score / 100) * availability_multiplier
    where fit_score (0-100) is gated on the four must-haves,
    and availability_multiplier (0.1-1.0) is an additive score clipped.

The fit_score is the dominant signal. Availability is a filter on top,
not a generator of high scores. A candidate with must_haves <= 1 will
have fit_score <= 25 and should never appear in the top 30, even with
availability_multiplier = 1.0.

ponytail: keep the formula simple. The complexity is in the components
(in must_haves.py), not the formula.
"""

from __future__ import annotations

from datetime import date

import config
import must_haves
from features import Features
from trap_detector import TrapInfo


# Fit score per must-have tier (from config.FIT_TIER_BASELINES).
NICE_TO_HAVE_BONUS_PER_HIT = 1.5  # max +10 (3 hits × 3 categories max)
NICE_TO_HAVE_BONUS_MAX = 10.0


def _evidence_quality_bonus(candidate: dict) -> float:
    """Return a small bonus [0, 20] based on strength of must-have evidence."""
    mh = must_haves.detect_must_haves(candidate)
    total = 0.0
    for info in mh.values():
        if info["met"]:
            total += 5 * info["n_primary"] + 2 * info["n_context"]
    return min(20.0, total)


def compute_fit_score(candidate: dict) -> tuple:
    """Return (fit_score, must_haves_met, evidence_dict).

    fit_score is 0-100, gated on the four must-haves per audit spec.
    """
    # Hard disqualifier check
    disq, disq_reason = must_haves.apply_hard_disqualifiers(candidate)
    if disq:
        return 0.0, 0, {"disqualified": disq_reason}

    # Detect must-haves
    mh = must_haves.detect_must_haves(candidate)
    n_met = sum(1 for v in mh.values() if v["met"])
    if n_met not in config.FIT_TIER_BASELINES:
        n_met = max(0, min(4, n_met))
    low, high = config.FIT_TIER_BASELINES[n_met]

    # Quality bonus maps evidence strength into the tier range.
    # quality_bonus=0 → low, quality_bonus=20 → high.
    quality_bonus = _evidence_quality_bonus(candidate)
    tier_width = high - low
    base = low + (tier_width * (quality_bonus / 20.0))

    # Nice-to-haves bonus (max +10)
    nth = must_haves.detect_nice_to_haves(candidate)
    nth_bonus = min(
        NICE_TO_HAVE_BONUS_MAX,
        sum(NICE_TO_HAVE_BONUS_PER_HIT * v for v in nth.values())
    )

    # Negative adjustments
    negatives = must_haves.detect_negative_patterns(candidate)
    neg_penalty = 0.0
    if "platform_team_handled" in negatives:
        neg_penalty += 15
    if "title_chasing" in negatives:
        neg_penalty += 8
    if "closed_source_only" in negatives:
        neg_penalty += 10
    if "explicit_gap_admission" in negatives:
        neg_penalty += 12

    # Outside-band YOE penalty
    profile = candidate.get("profile", {}) or {}
    yoe = profile.get("years_of_experience", 0) or 0
    if n_met <= 2 and (yoe < 5 or yoe > 9):
        neg_penalty += 5

    fit_score = base + nth_bonus - neg_penalty
    fit_score = max(0.0, min(100.0, fit_score))
    return fit_score, n_met, mh


def compute_availability_score(candidate: dict) -> float:
    """Additive availability score, clipped to [0.1, 1.0] per audit spec.

    Components (additive, then clipped):
    - open_to_work_flag: 1.0 if True, 0.25 if False (so contribution is 0.40 or 0.10)
    - notice_period_score: 1.0 at <=30d, decaying linearly to 0.2 at 120d+
    - recruiter_response_rate: 0.0-1.0 (raw)
    - recency: 1.0 if active in 30d, decaying linearly to 0 at 180d
    """
    rs = candidate.get("redrob_signals", {}) or {}

    # open_to_work_flag
    if rs.get("open_to_work_flag"):
        open_score = 1.0
    else:
        open_score = 0.25  # so contribution is 0.10 (0.25 * 0.40)

    # notice_period
    notice = rs.get("notice_period_days", 0) or 0
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 120:
        notice_score = 1.0 - (notice - 30) / 90 * 0.8
    else:
        notice_score = 0.2

    # recruiter_response_rate
    resp_score = max(0.0, min(1.0, rs.get("recruiter_response_rate", 0) or 0))

    # recency
    last_active = rs.get("last_active_date", "")
    recency_score = 0.0
    if last_active:
        try:
            d = date.fromisoformat(last_active)
            days = (date.today() - d).days
            if days <= 30:
                recency_score = 1.0
            elif days <= 180:
                recency_score = 1.0 - (days - 30) / 150
            else:
                recency_score = 0.0
        except (ValueError, TypeError):
            recency_score = 0.0

    total = (
        config.AVAILABILITY_WEIGHTS["open_to_work"] * open_score +
        config.AVAILABILITY_WEIGHTS["notice_period"] * notice_score +
        config.AVAILABILITY_WEIGHTS["recruiter_response"] * resp_score +
        config.AVAILABILITY_WEIGHTS["recency"] * recency_score
    )
    return max(0.1, min(1.0, total))


def compute_final_score(candidate: dict) -> dict:
    """Compute all scores for a candidate. Returns a dict with:
      - fit_score: 0-100
      - must_haves_met: 0-4
      - availability: 0.1-1.0
      - final_score: 0-1 (fit/100 * availability)
      - evidence: dict of must-have detection results
    """
    fit, n_met, evidence = compute_fit_score(candidate)
    avail = compute_availability_score(candidate)
    final = (fit / 100.0) * avail
    return {
        "fit_score": fit,
        "must_haves_met": n_met,
        "availability": avail,
        "final_score": final,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Legacy interface (kept for app.py and tests that still pass Features)
# ---------------------------------------------------------------------------

def compute_relevance(f: Features) -> float:
    """Legacy stub. The new model scores from raw candidate dict."""
    return 0.0


def compute_availability(f: Features) -> float:
    """Legacy stub. The new model scores from raw candidate dict."""
    return 0.0


def compute_score(f: Features, trap: TrapInfo) -> float:
    """Legacy interface. Use compute_final_score(candidate) instead."""
    if trap.is_honeypot:
        return config.HONEYPOT_TAX
    return 0.0
