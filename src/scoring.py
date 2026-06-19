"""scoring.py — Composite score from Features and trap multiplier.

The scoring formula is documented in config.py. This module is a pure
function: Features + TrapInfo → float score.

NEW FORMULA (per user feedback 2026-06-19):
    final_score = fit_score × availability_multiplier × trap_multiplier

The availability signals (open_to_work, notice_period, recruiter_response,
last_active) are no longer additive — they are a multiplicative filter.
This matches the JD: "a perfect-on-paper candidate who hasn't logged in
for 6 months and has a 5% recruiter response rate is, for hiring purposes,
not actually available. Down-weight them appropriately."

If availability is just 10% of an additive score, a candidate scoring
0.85 in fit × 0.5 in availability = 0.79 still ranks above a perfect-
fit candidate with no availability data. The multiplicative filter
ensures that the perfect-fit-with-no-availability candidate drops to
0.85 × 0.5 = 0.42 — clearly below an average-fit-and-available candidate
at 0.70 × 1.0 = 0.70.

ponytail: keep the formula simple. The complexity is in the
_components_ (in features.py), not the formula.
"""

from __future__ import annotations

import math

import config
from features import Features
from trap_detector import TrapInfo


def compute_score(f: Features, trap: TrapInfo) -> float:
    """Compute the composite score for one candidate.

    Returns:
        Final score in [0, ~1]. Honeypots return config.HONEYPOT_TAX (very
        negative) so they sort to the bottom of any ranking.
    """
    if trap.is_honeypot:
        return config.HONEYPOT_TAX

    fit = _fit_score(f, trap)
    avail = _availability_multiplier(f)
    return trap.trap_multiplier * fit * avail


def compute_relevance(f: Features) -> float:
    """JD-relevance score (Rule 8): how well does this candidate match the JD?

    Range [0, 1]. This is the fit score WITHOUT the availability multiplier.
    """
    return _fit_score(
        f,
        type("T", (), {
            "is_consulting_only": False,
            "is_keyword_stuffer": False,
            "is_template_summary": False,
            "is_title_chaser": False,
        })(),
    )


def compute_availability(f: Features) -> float:
    """Availability score: the multiplicative filter applied to fit.

    Range [0, 1]. Combines open_to_work, notice_period, recruiter_response,
    and last_active_date.
    """
    return _availability_multiplier(f)


def compute_authenticity(f: Features, trap: TrapInfo) -> float:
    """Authenticity score (Rule 8): does the profile make sense?

    Range [0, 1]. High authenticity = career evidence + product companies +
    no consulting-only history + no honeypot flags.
    """
    score = 1.0
    if trap.is_consulting_only:
        score *= 0.4
    if trap.is_keyword_stuffer:
        score *= 0.3
    if trap.is_template_summary:
        score *= 0.6
    if trap.is_title_chaser:
        score *= 0.7
    if f.has_ai_title_in_history:
        score = min(1.0, score * 1.1)
    return min(1.0, score)


def _fit_score(f: Features, trap: TrapInfo) -> float:
    """The technical fit score. Range [0, 1].

    Computed as a weighted sum of: career_history_relevance, project_impact,
    skills, company_quality, education. Availability is NOT in this score —
    it's a separate multiplicative filter.
    """
    weighted_sum = (
        config.WEIGHTS["career_history_relevance"] * _career_history_relevance(f) +
        config.WEIGHTS["project_impact"] * _project_impact(f) +
        config.WEIGHTS["skills"] * _skills_score(f) +
        config.WEIGHTS["company_quality"] * _company_quality_score(f, trap) +
        config.WEIGHTS["education"] * f.education_score
    )
    # Normalize by the new weights (which sum to 0.95 because availability
    # was removed from the additive sum and reallocated).
    total_weight = sum(
        v for k, v in config.WEIGHTS.items() if k != "availability"
    )
    return min(1.0, weighted_sum / total_weight)


def _availability_multiplier(f: Features) -> float:
    """Multiplicative availability filter. Range [0, 1].

    Per JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is, for hiring purposes, not
    actually available. Down-weight them appropriately."

    The four signals are MULTIPLIED together (not averaged) so that
    multiple weak signals compound. E.g., not_open_to_work (0.5) x 90d
    notice (0.5) x 5% response (0.525) = 0.13 — the candidate effectively
    drops out of the running.

    ponytail: the product of the four components is bounded by [0, 1] and
    monotonically decreasing as any one signal gets worse. This is the
    right shape for "all of these must be at least OK" semantics.
    """
    # 1. open_to_work_flag: binary, hard hit if False.
    # JD: "open_to_work" is the strongest explicit signal of intent.
    open_mult = 1.0 if f.open_to_work else 0.5
    # 2. notice_period_days: JD wants sub-30 days; 30+ faces a higher bar.
    notice = f.notice_period_days
    if notice <= 0:
        notice_mult = 1.0   # immediate — perfect
    elif notice <= 30:
        notice_mult = 1.0   # sub-30-day — perfect (per JD)
    elif notice <= 60:
        notice_mult = 0.85  # OK
    elif notice <= 90:
        notice_mult = 0.65  # borderline
    elif notice <= 120:
        notice_mult = 0.45  # bad
    else:  # >120
        notice_mult = 0.30  # very bad
    # 3. recruiter_response_rate: how often the candidate replies to
    # recruiters. 0% -> 0.5, 100% -> 1.0. JD: "5% recruiter response rate
    # is, for hiring purposes, not actually available".
    resp_rate = f.recruiter_response_rate
    resp_mult = 0.5 + 0.5 * max(0.0, min(1.0, resp_rate))
    # 4. days_since_active: last login recency. JD: "hasn't logged in
    # for 6 months ... is not actually available". 999 means "never
    # active" or "missing" — treat as very bad.
    days = f.days_since_active
    if days <= 30:
        active_mult = 1.0    # active this month
    elif days <= 90:
        active_mult = 0.90   # active in last quarter
    elif days <= 180:
        active_mult = 0.65   # active in last 6 months
    elif days <= 365:
        active_mult = 0.40   # last year
    elif days == 999:
        active_mult = 0.30   # never / missing data
    else:
        active_mult = 0.20   # very stale

    product = open_mult * notice_mult * resp_mult * active_mult
    return min(1.0, max(0.0, product))


def _career_history_relevance(f: Features) -> float:
    """Career history relevance: biggest signal in fit score.

    Per Rule 1: 'A candidate who built ranking systems should outrank a
    candidate who merely knows Pinecone.' Per Rule 3: 'Career evidence
    always beats skills.' We reward:
    - AI/ML titles in history (foundational)
    - High-value roles (search/retrieval/ranking/rec/relevance/matching)
    - AI keyword hits in descriptions
    - Production experience (deploy/serving/shipped)
    """
    score = 0.0
    if f.has_ai_title_in_history:
        score += 0.30
    if f.high_value_role_count > 0:
        score += 0.40 * min(1.0, math.log1p(f.high_value_role_count) / math.log1p(3))
    if f.career_ai_keyword_hits > 0:
        score += 0.20 * min(1.0, math.log1p(f.career_ai_keyword_hits) / math.log1p(4))
    # Production evidence
    production_keywords = ["deploy", "deployed", "production", "serving", "shipped", "ship", "scaled"]
    hits = 0
    for c in f.career_history:
        desc = (c.get("description") or "").lower()
        for kw in production_keywords:
            if kw in desc:
                hits += 1
                break
    if hits > 0:
        score += 0.10 * min(1.0, math.log1p(hits) / math.log1p(3))
    return min(1.0, score)


def _project_impact(f: Features) -> float:
    """Project impact score in fit.

    Per Rule 5: 'Evaluation expertise is a major ranking factor.' Per
    Rule 6: 'Search/retrieval/ranking/recommendation/relevance/matching/
    personalization experience are the highest-value signals.'

    Returns [0, 1]. Each evidence category contributes its share.
    """
    if not f.project_impact_counts:
        return 0.0
    score = 0.0
    for category, weight in config.PROJECT_IMPACT_WEIGHTS.items():
        if f.project_impact_counts.get(category, 0) > 0:
            score += weight * min(1.0, f.project_impact_counts[category] / 2.0)
    return min(1.0, score)


def _skills_score(f: Features) -> float:
    """Skills score in fit.

    Per Rule 3: 'Career evidence always beats skills.' Skills are
    supporting signals, not the primary driver.
    """
    score = f.ai_skills_depth
    if f.has_foundational_ml:
        score = min(1.0, score + 0.10)
    if f.has_production_ai_tools:
        score = min(1.0, score + 0.05)
    return score


def _company_quality_score(f: Features, trap: TrapInfo) -> float:
    """Company quality score in fit.

    Combines product vs consulting split, location, and (negatively) the
    consulting-only trap.
    """
    score = f.product_exp_score
    if trap.is_consulting_only:
        score *= 0.5
    score = 0.7 * score + 0.3 * f.location_score
    return min(1.0, score)


def _jd_must_have_score(f: Features) -> float:
    """Explicit score for matching the JD's specific must-haves.

    Used inside project_impact for eval/relevance evidence. Range [0, 1].
    """
    n_must_haves = len(f.matched_must_haves)
    if n_must_haves >= 4:
        return 1.0
    if n_must_haves == 3:
        return 0.75
    if n_must_haves == 2:
        return 0.5
    if n_must_haves == 1:
        return 0.25
    return 0.0
