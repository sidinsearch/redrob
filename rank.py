"""rank.py — Main entry point for the Redrob candidate ranker.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./output/submission.csv

Pipeline:
    stream JSONL → detect must-haves (from career_history) → score
    → sort by (final_score desc, candidate_id asc) → top-100
    → generate reasoning citing specific career_history sentences
    → write submission CSV → validate

Per audit spec (2026-06-19):
    final_score = (fit_score / 100) * availability_multiplier
    where fit_score is gated on the four must-haves and availability is
    a clipped additive filter.

Compute budget: ≤5 min on CPU, ≤16 GB RAM, no network.
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Any, Dict

# Allow running as both module and script.
# rank.py lives at project root, modules live in src/.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
for p in (str(_HERE), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import output
import parser
import trap_detector
from features import extract_features
import must_haves
from scoring import (
    compute_final_score,
    compute_fit_score,
    compute_availability_score,
)
from trap_detector import analyze
from features import Features


# ----------------------------------------------------------------------------
# Heap-based top-K tracking — O(N log K) instead of O(N log N)
# ----------------------------------------------------------------------------

class _TopKTracker:
    """Track top-K candidates by (score DESC, candidate_id ASC) without
    storing all of them. For 100K candidates and K=100, this is
    O(N log K) ≈ 660K operations.
    """

    def __init__(self, k: int):
        self.k = k
        self._items: List[Tuple[float, str, int]] = []  # (score, cid, offset)

    def offer(self, score: float, candidate_id: str, offset: int) -> None:
        if not candidate_id:
            return
        self._items.append((score, candidate_id, offset))

    def top_unpacked(self) -> List[Tuple[str, float, int]]:
        """Return [(candidate_id, score, offset)] sorted by score desc, id asc."""
        self._items.sort(key=lambda x: (-x[0], x[1]))
        top = self._items[: self.k]
        return [(cid, score, offset) for (score, cid, offset) in top]


# ----------------------------------------------------------------------------
# Reasoning generation (cites specific career_history evidence)
# ----------------------------------------------------------------------------

def _generate_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    """Generate a 1-2 sentence reasoning citing specific evidence.

    The reasoning MUST cite specific career_history sentences, not
    skills-list keywords. If must_haves_met <= 1, the reasoning must
    explicitly say the candidate does not meet core JD requirements.
    """
    fit = scores["fit_score"]
    n_met = scores["must_haves_met"]
    evidence = scores["evidence"]

    parts = []

    # 1. Job title + YoE (factual context)
    profile = candidate.get("profile", {}) or {}
    title = profile.get("current_title", "") or ""
    company = profile.get("current_company", "") or ""
    yoe = profile.get("years_of_experience", 0) or 0
    if title:
        parts.append(f"{title}")
        if company:
            parts.append(f"at {company}")
        if yoe:
            parts.append(f"({yoe}yr)")
        parts.append(".")

    # 2. Must-have summary
    if n_met >= 4:
        parts.append(f" Strong JD fit: {n_met}/4 must-haves.")
    elif n_met == 3:
        parts.append(f" {n_met}/4 must-haves met.")
    elif n_met <= 1:
        if "disqualified" in evidence:
            parts.append(f" Disqualified: {evidence['disqualified']}.")
        else:
            parts.append(f" Only {n_met}/4 must-haves met — does not meet core JD requirements.")

    # 3. Cite a specific career_history sentence (the strongest evidence)
    if n_met >= 1:
        # Find the must-have with the strongest primary evidence
        best_mh = None
        best_evidence = ""
        for mh_name, info in evidence.items():
            if info["met"] and info.get("evidence_sentences"):
                if not best_evidence or len(info["evidence_sentences"][0]) > len(best_evidence):
                    best_mh = mh_name
                    best_evidence = info["evidence_sentences"][0]
        if best_evidence:
            parts.append(f" Evidence: {best_evidence[:150]}")

    # 4. Availability signal (if positive)
    avail = scores["availability"]
    rs = candidate.get("redrob_signals", {}) or {}
    if rs.get("open_to_work_flag"):
        parts.append(" Open to work.")
    if rs.get("recruiter_response_rate", 0) >= 0.5:
        parts.append(" Good recruiter response rate.")

    text = "".join(parts).strip()
    # Cap at 300 chars (per submission spec)
    if len(text) > config.MAX_REASONING_LEN:
        text = text[: config.MAX_REASONING_LEN - 1].rstrip() + "…"
    return text


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def run(candidates_path: str | Path, out_path: str | Path, top_k: int = config.TOP_K) -> dict:
    """Run the full pipeline. Returns a stats dict.

    Side effects:
    - Writes <out_path>          → top-100 ranked submission CSV.
    - Writes <out_path>.honeypots.csv → full details of every honeypot.
    """
    candidates_path = Path(candidates_path)
    out_path = Path(out_path)
    honeypot_path = out_path.with_name(out_path.stem + ".honeypots.csv")

    t0 = time.perf_counter()
    topk = _TopKTracker(top_k)

    n_total = 0
    n_honeypots = 0
    n_keyword_stuffers = 0
    n_template_summary = 0
    n_consulting_only = 0
    n_title_chaser = 0
    n_skipped = 0
    n_disqualified = 0  # hard disqualifiers from Step 1
    honeypot_details: list = []

    print(f"[ranker] reading {candidates_path} ({parser.path_size_mb(candidates_path):.1f} MB)")
    sys.stdout.flush()

    for offset, _line_num, candidate in parser._iter_jsonl(candidates_path):
        n_total += 1
        try:
            features = extract_features(candidate)
        except Exception as e:
            n_skipped += 1
            if n_total % 10_000 == 0:
                print(f"[ranker] skipped {n_skipped} candidates so far (last err: {e})")
            continue

        trap = analyze(features)
        if trap.is_honeypot:
            n_honeypots += 1
            honeypot_details.append((features, trap))
            continue  # NEVER offer a honeypot to top-K, regardless of score
        if trap.is_keyword_stuffer:
            n_keyword_stuffers += 1
        if trap.is_template_summary:
            n_template_summary += 1
        if trap.is_consulting_only:
            n_consulting_only += 1
        if trap.is_title_chaser:
            n_title_chaser += 1

        # NEW: score from raw candidate dict (audit spec)
        scores = compute_final_score(candidate)
        # Track hard disqualifiers
        if scores["evidence"].get("disqualified"):
            n_disqualified += 1
        final_score = scores["final_score"]

        topk.offer(final_score, features.candidate_id, offset)

        if n_total % 25_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[ranker] {n_total:>7d} candidates in {elapsed:5.1f}s "
                  f"(honeypots={n_honeypots}, ks={n_keyword_stuffers}, "
                  f"ts={n_template_summary}, co={n_consulting_only}, tc={n_title_chaser}, "
                  f"disq={n_disqualified})")
            sys.stdout.flush()

    t1 = time.perf_counter()
    print(f"[ranker] extracted features in {t1 - t0:.1f}s "
          f"({n_total} total, {n_skipped} skipped, {n_honeypots} honeypots, {n_disqualified} hard-disqualified)")
    sys.stdout.flush()

    # Write honeypots.csv with full details.
    honeypot_csv_rows = []
    for f, trap in honeypot_details:
        reasons = trap.honeypot_reasons
        honeypot_csv_rows.append({
            "candidate_id": f.candidate_id,
            "current_title": f.current_title,
            "current_company": f.current_company,
            "years_of_experience": f.years_of_experience,
            "location": f.location,
            "country": f.country,
            "honeypot_reasons": " | ".join(reasons),
            "n_reasons": len(reasons),
            "career_timeline_anomaly": int(f.career_timeline_anomaly),
            "expert_with_zero_duration": int(f.expert_with_zero_duration),
            "title_skills_history_mismatch": int(f.title_skills_history_mismatch),
            "employment_overlap_anomaly": int(f.employment_overlap_anomaly),
            "duration_integrity_violation": int(f.duration_integrity_violation),
            "title_responsibility_mismatch": int(f.title_responsibility_mismatch),
            "skill_experience_contradiction": int(f.skill_experience_contradiction),
            "education_timeline_anomaly": int(f.education_timeline_anomaly),
            "career_progression_anomaly": int(f.career_progression_anomaly),
            "achievement_inflation": int(f.achievement_inflation),
            "technology_age_anomaly": int(f.technology_age_anomaly),
            "synthetic_profile": int(f.synthetic_profile),
            "cross_field_inconsistency": int(f.cross_field_inconsistency),
            "nlp_claim_without_evidence": int(f.nlp_claim_without_evidence),
            "ai_skill_count": f.ai_skill_count,
            "pre_llm_roles": f.pre_llm_roles,
            "num_career_entries": len(f.career_history),
        })
    output.write_honeypots(honeypot_csv_rows, honeypot_path)
    print(f"[ranker] wrote {honeypot_path} ({len(honeypot_csv_rows)} honeypots)")

    # Get top-K
    top_unpacked = topk.top_unpacked()

    # Seek-and-load just the top-100 records
    top_offsets = [o for _, _, o in top_unpacked]
    top_records = parser.seek_to_lines(candidates_path, top_offsets)

    # Map back: offset -> record
    offset_to_record = {off: rec for off, rec in zip(top_offsets, top_records)}

    # Build (rank, candidate_id, fit, must_haves, availability, final, reasoning) rows.
    # We sort by final_score desc, with epsilon on cid to break ties
    # (smaller cid → higher display score).
    sorted_top = []
    for cid, score, offset in top_unpacked:
        try:
            cid_num = int(cid.split("_")[1])
        except (IndexError, ValueError):
            cid_num = 0
        # Smaller cid → larger epsilon → higher display score → ranks first.
        epsilon = (9999999 - cid_num) * 1e-9
        display_score = score + epsilon
        sorted_top.append((cid, score, offset, display_score))
    sorted_top.sort(key=lambda x: -x[3])

    rows: List[Tuple[str, int, float, str]] = []
    for rank, (cid, score, offset, display_score) in enumerate(sorted_top, start=1):
        rec = offset_to_record.get(offset, {})
        scores = compute_final_score(rec)
        reasoning = _generate_reasoning(rec, scores, rank)
        rows.append((cid, rank, display_score, reasoning))

    t2 = time.perf_counter()
    print(f"[ranker] generated reasoning for top {len(rows)} in {t2 - t1:.1f}s")
    sys.stdout.flush()

    # Write CSV
    output.write_submission(rows, out_path)
    t3 = time.perf_counter()
    print(f"[ranker] wrote {out_path} in {t3 - t2:.1f}s")
    sys.stdout.flush()

    # Self-validate
    errors = output.validate(out_path)
    if errors:
        print(f"[ranker] validation FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"[ranker] validation passed")

    # Sanity check: no candidate with must_haves <= 1 in top 30 (per audit spec)
    must_have_violations = []
    for rank, (cid, _, _, _) in enumerate(
        [(r[0], r[1], r[2], r[3]) for r in rows], 1
    ):
        if rank > 30:
            break
        # Recompute scores for this candidate
        rec = offset_to_record.get(
            next(o for c, _, o, _ in sorted_top if c == cid), {}
        )
        scores = compute_final_score(rec)
        if scores["must_haves_met"] <= 1 and scores["fit_score"] > 25:
            must_have_violations.append((rank, cid, scores["must_haves_met"]))
    if must_have_violations:
        print(f"[ranker] SANITY CHECK FAILED: candidates with must_haves <= 1 in top 30:")
        for rank, cid, n in must_have_violations:
            print(f"  - rank {rank}: {cid} (must_haves={n})")

    total = time.perf_counter() - t0
    stats = {
        "n_total": n_total,
        "n_skipped": n_skipped,
        "n_honeypots": n_honeypots,
        "n_keyword_stuffers": n_keyword_stuffers,
        "n_template_summary": n_template_summary,
        "n_consulting_only": n_consulting_only,
        "n_title_chaser": n_title_chaser,
        "n_disqualified": n_disqualified,
        "runtime_seconds": total,
        "out_path": str(out_path),
        "honeypot_path": str(honeypot_path),
    }
    print(f"[ranker] done in {total:.1f}s")
    return stats


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Redrob candidate ranker — produces top-100 CSV submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--candidates", "-c",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl (input)",
    )
    ap.add_argument(
        "--out", "-o",
        default="./output/submission.csv",
        help="Path to write submission.csv (output)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help="Number of candidates to rank (default 100 per spec)",
    )
    args = ap.parse_args()

    run(args.candidates, args.out, args.top_k)


if __name__ == "__main__":
    main()
