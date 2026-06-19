"""scoring.py — Composite score from Features and trap multiplier.

The scoring formula is documented in config.py. This module is a pure
function: Features + TrapInfo → float score.

Final score = trap_multiplier * weighted_sum_of_components

We rank in float; ties broken by candidate_id ascending per spec.

ponytail: scoring is a single linear combination. The complexity is in
the *components* (in features.py), not in the formula. Keep this simple.

We separate RELEVANCE from AUTHENTICITY per Rule 8:
- relevance_score: does the candidate match the JD?
- authenticity_score: does the profile make sense?
- final_score = relevance * authenticity
Honeypots and consulting-only profiles have low authenticity.
"""

from __future__ import annotations

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

    weighted_sum = (
        config.WEIGHTS["career_history_relevance"] * _career_history_relevance(f) +
        config.WEIGHTS["project_impact"] * _project_impact(f) +
        config.WEIGHTS["skills"] * _skills_score(f) +
        config.WEIGHTS["availability"] * _availability_score(f) +
        config.WEIGHTS["company_quality"] * _company_quality_score(f, trap) +
        config.WEIGHTS["education"] * f.education_score
    )

    return trap.trap_multiplier * weighted_sum


def compute_relevance(f: Features) -> float:
    """JD-relevance score (Rule 8): how well does this candidate match the JD?

    Range [0, 1]. Used to separate 'highly relevant but suspicious' from
    'authentic but not relevant'. This is the raw score WITHOUT the
    authenticity multiplier.
    """
    if not f.career_history and f.ai_skill_count == 0:
        return 0.0
    return (
        config.WEIGHTS["career_history_relevance"] * _career_history_relevance(f) +
        config.WEIGHTS["project_impact"] * _project_impact(f) +
        config.WEIGHTS["skills"] * _skills_score(f) +
        config.WEIGHTS["availability"] * _availability_score(f) +
        config.WEIGHTS["company_quality"] * _company_quality_score(
            f, type("T", (), {"is_consulting_only": False, "is_keyword_stuffer": False, "is_template_summary": False})()
        ) +
        config.WEIGHTS["education"] * f.education_score
    )


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
        score = min(1.0, score * 1.1)  # small boost for real AI history
    return min(1.0, score)


def _career_history_relevance(f: Features) -> float:
    """Career history relevance: biggest signal (40% weight).

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
        # log-normalize: 1 role = 0.20, 2 roles = 0.32, 3+ = 0.40
        import math
        score += 0.40 * min(1.0, math.log1p(f.high_value_role_count) / math.log1p(3))
    if f.career_ai_keyword_hits > 0:
        import math
        score += 0.20 * min(1.0, math.log1p(f.career_ai_keyword_hits) / math.log1p(4))
    # Production evidence in any description
    production_keywords = ["deploy", "deployed", "production", "serving", "shipped", "ship", "scaled"]
    hits = 0
    for c in f.career_history:
        desc = (c.get("description") or "").lower()
        for kw in production_keywords:
            if kw in desc:
                hits += 1
                break
    if hits > 0:
        import math
        score += 0.10 * min(1.0, math.log1p(hits) / math.log1p(3))
    return min(1.0, score)


def _project_impact(f: Features) -> float:
    """Project impact score: 20% weight.

    Per Rule 5: 'Evaluation expertise is a major ranking factor.' Per
    Rule 6: 'Search/retrieval/ranking/recommendation/relevance/matching/
    personalization experience are the highest-value signals.' Per user
    spec: 'if candidate deployed retrieval: +25', 'if candidate built
    ranking: +25', 'if candidate improved NDCG: +20', 'if candidate ran
    A/B tests: +15'.

    Returns [0, 1]. Each evidence category contributes its share; the
    sum is the project_impact score.
    """
    if not f.project_impact_counts:
        return 0.0
    score = 0.0
    for category, weight in config.PROJECT_IMPACT_WEIGHTS.items():
        if f.project_impact_counts.get(category, 0) > 0:
            # Cap each category at 1.0 (so multiple hits don't overweight).
            score += weight * min(1.0, f.project_impact_counts[category] / 2.0)
    return min(1.0, score)


def _skills_score(f: Features) -> float:
    """Skills score: 15% weight.

    Per Rule 3: 'Career evidence always beats skills.' Skills are
    supporting signals, not the primary driver. We use ai_skills_depth
    and foundational-ml presence as a sanity check, but a candidate with
    strong skills but no career evidence will not rank high overall.
    """
    score = f.ai_skills_depth  # already in [0, 1]
    if f.has_foundational_ml:
        score = min(1.0, score + 0.10)
    if f.has_production_ai_tools:
        score = min(1.0, score + 0.05)
    return score


def _availability_score(f: Features) -> float:
    """Availability score: 10% weight.

    Combines notice period + open-to-work + recruiter engagement.
    Per Rule 7: 'Availability and recruiter engagement signals matter.'
    """
    base = f.availability_score  # notice period score in [0, 1]
    if f.open_to_work:
        base = min(1.0, base + 0.20)
    if f.search_appearance_30d > 0:
        # log-normalize: log1p(50) / log1p(500) ~= 0.5
        import math
        base = min(1.0, base + 0.20 * min(1.0, math.log1p(f.search_appearance_30d) / math.log1p(500)))
    if f.saved_by_recruiters_30d > 0:
        import math
        base = min(1.0, base + 0.15 * min(1.0, math.log1p(f.saved_by_recruiters_30d) / math.log1p(30)))
    # Behavioral signals summary
    base = min(1.0, base + 0.10 * f.behavioral_score)
    return base


def _company_quality_score(f: Features, trap: TrapInfo) -> float:
    """Company quality: 10% weight.

    Combines product vs consulting split, location, and (negatively) the
    consulting-only trap. Per JD: 'If you're currently at one of these
    companies but have prior product-company experience, that's fine.'
    """
    score = f.product_exp_score
    # Apply consulting penalty
    if trap.is_consulting_only:
        score *= 0.5
    # Location score
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
