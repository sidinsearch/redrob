"""ranking_pipeline.py — Reusable ranking pipeline for the Streamlit sandbox.

Extracts the core ranking logic from rank.py so the Streamlit app can call
it on a list of dicts (no need for file paths). The full-file ranker in
rank.py uses this same code with file streaming.
"""

from __future__ import annotations

from typing import List, Tuple

import config
from features import extract_features
from reasoning import generate_reasoning
from scoring import compute_score
from trap_detector import analyze


def rank_candidates(
    candidates: List[dict], top_k: int = config.TOP_K
) -> Tuple[List[Tuple[str, int, float, str]], dict]:
    """Rank a list of candidate dicts. Returns (rows, trap_stats).

    rows: list of (candidate_id, rank, score, reasoning) tuples
    trap_stats: dict with counts of each trap type
    """
    scored: List[Tuple[float, str, dict]] = []  # (score, cid, candidate_dict)
    trap_stats = {
        "total_honeypots": 0,
        "total_keyword_stuffers": 0,
        "total_template_summary": 0,
        "total_consulting_only": 0,
        "total_title_chaser": 0,
        "total_clean": 0,
        "in_topk": 0,
    }

    for cand in candidates:
        f = extract_features(cand)
        t = analyze(f)
        if t.is_honeypot:
            trap_stats["total_honeypots"] += 1
        if t.is_keyword_stuffer:
            trap_stats["total_keyword_stuffers"] += 1
        if t.is_template_summary:
            trap_stats["total_template_summary"] += 1
        if t.is_consulting_only:
            trap_stats["total_consulting_only"] += 1
        if t.is_title_chaser:
            trap_stats["total_title_chaser"] += 1
        if not any([t.is_honeypot, t.is_keyword_stuffer, t.is_template_summary,
                    t.is_consulting_only, t.is_title_chaser]):
            trap_stats["total_clean"] += 1

        score = compute_score(f, t)
        scored.append((score, cand.get("candidate_id", ""), cand))

    # Sort by score desc, candidate_id asc
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:top_k]

    # Build output rows with reasoning
    rows = []
    for rank, (score, cid, cand) in enumerate(top, start=1):
        f = extract_features(cand)
        t = analyze(f)
        if t.is_honeypot:
            trap_stats["in_topk"] += 1
        reasoning = generate_reasoning(f, t, rank)
        rows.append((cid, rank, score, reasoning))

    return rows, trap_stats
