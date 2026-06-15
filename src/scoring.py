"""scoring.py — Composite score from Features and trap multiplier.

The scoring formula is documented in config.py. This module is a pure
function: Features + TrapInfo → float score.

Final score = trap_multiplier * weighted_sum_of_components

We rank in float; ties broken by candidate_id ascending per spec.

ponytail: scoring is a single linear combination. The complexity is in
the *components* (in features.py), not in the formula. Keep this simple.
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
        config.WEIGHTS["title_relevance"] * f.title_score +
        config.WEIGHTS["experience_fit"] * f.experience_fit +
        config.WEIGHTS["product_exp"] * f.product_exp_score +
        config.WEIGHTS["ai_skills_depth"] * f.ai_skills_depth +
        config.WEIGHTS["career_relevance"] * _career_relevance(f) +
        config.WEIGHTS["education_score"] * f.education_score +
        config.WEIGHTS["behavioral_score"] * f.behavioral_score +
        config.WEIGHTS["location_fit"] * f.location_score +
        config.WEIGHTS["availability_score"] * f.availability_score
    )

    return trap.trap_multiplier * weighted_sum


def _career_relevance(f: Features) -> float:
    """Career relevance sub-score: AI titles + keyword hits in descriptions.

    A candidate with "ML Engineer" in their history + 3 jobs with retrieval
    keywords scores higher than someone with no AI history.
    """
    score = 0.0
    if f.has_ai_title_in_history:
        score += 0.5
    if f.career_ai_keyword_hits > 0:
        # Log-normalize hits
        import math
        score += 0.3 * min(1.0, math.log1p(f.career_ai_keyword_hits) / math.log1p(4))
    # Project references (heuristic: presence of "deploy", "serving", "production")
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
        score += 0.2 * min(1.0, math.log1p(hits) / math.log1p(3))
    return min(1.0, score)
