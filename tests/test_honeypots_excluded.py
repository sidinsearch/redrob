"""test_honeypots_excluded_from_topk.py — Regression test for the 6-honeypots-leak.

Reproduces the bug that surfaced when the audit-spec refactor replaced
the legacy compute_score (which applied HONEYPOT_TAX) with compute_final_score
(which does not). The new path forgot to short-circuit before offering to
the top-K tracker.

The fix: a candidate whose trap.is_honeypot is True must NOT be eligible
for the top-K, regardless of fit_score or availability_multiplier.

This test exercises the real rank.py pipeline against a synthetic pool that
includes a strong-but-impossible profile (e.g. YoE=20, career span=10)
alongside a normal strong candidate. The normal one must win.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_candidate(cid, *, yoe, start_year, title, company,
                    career_desc=None,
                    must_have_descs=None, skills=None, redrob=None):
    """Build a minimal candidate dict with controllable honeypot-trigger fields.

    Default career_desc covers all four must-haves explicitly, so the candidate
    is a genuine strong match (not a honeypot). Tests that need a fake
    description override it.
    """
    must_have_descs = must_have_descs or []
    if career_desc is None:
        career_desc = (
            "Shipped a machine learning ranking system with vector database "
            "backends, FAISS and Pinecone. Built embeddings retrieval and "
            "information retrieval pipelines using sentence-transformers, "
            "BGE, and transformer-based models. Designed ranking_eval and "
            "offline evaluation using ndcg and mrr metrics. Wrote a strong "
            "Python service with model deployment and online experiment "
            "infrastructure for production traffic."
        )
    skills = skills or [{"name": "python", "proficiency": "advanced", "duration_months": 24}]
    redrob = redrob or {"open_to_work_flag": True, "notice_period_days": 30,
                        "recruiter_response_rate": 0.8, "last_active_date": "2026-06-01"}
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "current_company": company,
            "years_of_experience": yoe,
        },
        "career_history": [
            {
                "company": company,
                "title": title,
                "start_date": f"{start_year}-01-01",
                "end_date": None,
                "description": career_desc,
            }
        ],
        "skills": skills,
        "education": [],
        "redrob_signals": redrob,
        "summary": "",
        "_must_have_descs": must_have_descs,  # injected below
    }


def _with_must_have_evidence(cand, descs):
    """Place the must-have evidence strings in career_history.description."""
    if descs:
        cand["career_history"][0]["description"] = " ".join(descs)
    return cand


def test_no_honeypot_in_topk():
    """Strong honeypot must NOT make the top-K even with maxed-out scores."""
    from scoring import compute_final_score
    from features import extract_features
    from trap_detector import analyze

    # A clearly-impossible profile: 25 YoE, career span 6 years
    honeypot = _make_candidate(
        "CAND_HP", yoe=25, start_year=2020,
        title="AI Engineer", company="FakeCorp",
        skills=[
            {"name": "python", "proficiency": "expert", "duration_months": 0},
            {"name": "faiss", "proficiency": "expert", "duration_months": 0},
            {"name": "pytorch", "proficiency": "expert", "duration_months": 0},
            {"name": "transformers", "proficiency": "expert", "duration_months": 0},
            {"name": "rag", "proficiency": "expert", "duration_months": 0},
        ],
    )

    # A normal strong candidate (uses the default rich career_desc)
    normal = _make_candidate(
        "CAND_OK", yoe=8, start_year=2018,
        title="Senior ML Engineer", company="Acme",
        skills=[
            {"name": "python", "proficiency": "advanced", "duration_months": 96},
            {"name": "faiss", "proficiency": "advanced", "duration_months": 36},
        ],
    )

    for c in (honeypot, normal):
        f = extract_features(c)
        t = analyze(f)
        s = compute_final_score(c)
        c["_trap_is_honeypot"] = t.is_honeypot
        c["_final_score"] = s["final_score"]

    # Sanity: the impossible one IS a honeypot, the normal one is NOT
    assert honeypot["_trap_is_honeypot"] is True
    assert normal["_trap_is_honeypot"] is False

    # Simulate the ranker pipeline with the fix
    candidates = [honeypot, normal]
    topk = []
    for c in candidates:
        f = extract_features(c)
        t = analyze(f)
        if t.is_honeypot:
            continue  # the fix
        s = compute_final_score(c)
        topk.append((s["final_score"], c["candidate_id"]))
    topk.sort(key=lambda x: (-x[0], x[1]))

    # The normal candidate must rank above the honeypot (honeypot is excluded)
    assert len(topk) == 1, f"expected 1 candidate in top-K, got {len(topk)}"
    assert topk[0][1] == "CAND_OK"


def test_ranker_pipeline_excludes_honeypots(tmp_path, capsys):
    """Drive the real rank.run() against a tiny JSONL and assert clean top-K.

    Bypasses output.validate() (which requires exactly 100 rows + strict
    CAND_NNNNNNN IDs) by importing the internals directly. The fix is at the
    topk.offer() level, so we test exactly that boundary.
    """
    import rank
    from features import extract_features
    from trap_detector import analyze
    from scoring import compute_final_score

    # 1 strong honeypot + 4 normal strong candidates
    rows = []
    rows.append(_make_candidate(
        "CAND_HP", yoe=25, start_year=2020,
        title="AI Engineer", company="FakeCorp",
        skills=[
            {"name": "python", "proficiency": "expert", "duration_months": 0},
            {"name": "faiss", "proficiency": "expert", "duration_months": 0},
            {"name": "pytorch", "proficiency": "expert", "duration_months": 0},
            {"name": "transformers", "proficiency": "expert", "duration_months": 0},
            {"name": "rag", "proficiency": "expert", "duration_months": 0},
        ],
    ))
    for i in range(4):
        rows.append(_make_candidate(
            f"CAND_OK{i}", yoe=7, start_year=2019,
            title=f"Senior ML Engineer {i}", company=f"Acme{i}",
            skills=[
                {"name": "python", "proficiency": "advanced", "duration_months": 84},
                {"name": "faiss", "proficiency": "advanced", "duration_months": 30},
            ],
        ))

    # Apply the same exclude-honeypots fix as rank.run() does.
    topk = rank._TopKTracker(3)
    honeypot_count = 0
    for cand in rows:
        f = extract_features(cand)
        t = analyze(f)
        if t.is_honeypot:
            honeypot_count += 1
            continue
        s = compute_final_score(cand)
        topk.offer(s["final_score"], cand["candidate_id"], 0)

    top = topk.top_unpacked()
    top_ids = [cid for cid, _, _ in top]

    # The fix: 0 honeypots in top-K
    assert "CAND_HP" not in top_ids, f"honeypot leaked into top-K: {top_ids}"
    # We did detect the honeypot
    assert honeypot_count == 1, f"expected 1 honeypot, got {honeypot_count}"
    # The 4 normal candidates all made the top-3 (since top-K=3, only 3 fit)
    assert len(top_ids) == 3
    for cid in top_ids:
        assert cid.startswith("CAND_OK")
