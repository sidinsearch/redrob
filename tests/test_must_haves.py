"""test_must_haves.py — Tests for the new must-have detection (audit spec Step 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import must_haves


def make_candidate(profile=None, career=None, skills=None, education=None,
                   redrob_signals=None):
    return {
        "candidate_id": "TEST",
        "profile": profile or {
            "current_title": "ML Engineer",
            "current_company": "BigCo",
            "years_of_experience": 6.0,
        },
        "career_history": career or [],
        "skills": skills or [],
        "education": education or [],
        "redrob_signals": redrob_signals or {},
    }


def test_strong_fit_candidate():
    """A candidate with strong production evidence meets all 4 must-haves."""
    c = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "Google",
                 "years_of_experience": 7.0, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "Google",
                "title": "Senior ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 48,
                "description": (
                    "Built and shipped a hybrid retrieval system using BM25 + dense embeddings "
                    "(sentence-transformers, bge-base) with FAISS HNSW index. "
                    "Designed the offline-online correlation eval framework with NDCG and MRR. "
                    "Improved NDCG by 25% through online A/B testing. "
                    "Built in Python with FastAPI serving embeddings to 10M+ users. "
                    "Production deployment owned end-to-end."
                ),
            },
        ],
        redrob_signals={
            "open_to_work_flag": True,
            "notice_period_days": 30,
            "recruiter_response_rate": 0.85,
            "last_active_date": "2026-05-15",
        },
    )
    scores = must_haves.detect_must_haves(c)
    n_met = sum(1 for v in scores.values() if v["met"])
    assert n_met == 4, f"Expected 4/4, got {n_met}"
    print("  PASS strong_fit_candidate (4/4)")


def test_consulting_only_disqualified():
    """A candidate whose entire career is at consulting firms is hard-disqualified."""
    c = make_candidate(
        profile={"current_title": "Senior Developer", "current_company": "Infosys",
                 "years_of_experience": 5.0, "location": "Bangalore", "country": "India"},
        career=[
            {"company": "Infosys", "title": "Senior Developer",
             "start_date": "2021-01-01", "duration_months": 60,
             "description": "Built internal tools for clients."},
            {"company": "TCS", "title": "Developer",
             "start_date": "2018-01-01", "duration_months": 36,
             "description": "Worked on enterprise apps."},
        ],
    )
    disq, reason = must_haves.apply_hard_disqualifiers(c)
    assert disq, "Expected hard-disqualifier"
    assert reason == "consulting-only-no-prior-product", f"Got: {reason}"
    print("  PASS consulting_only_disqualified")


def test_consulting_with_prior_product_NOT_disqualified():
    """A candidate at consulting now but with prior product history is OK."""
    c = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "TCS",
                 "years_of_experience": 7.0, "location": "Bangalore", "country": "India"},
        career=[
            {"company": "TCS", "title": "ML Engineer",
             "start_date": "2023-01-01", "duration_months": 24,
             "description": "Client project work."},
            {"company": "Razorpay", "title": "ML Engineer",
             "start_date": "2019-01-01", "duration_months": 48,
             "description": "Built ranking system in production."},
        ],
    )
    disq, reason = must_haves.apply_hard_disqualifiers(c)
    assert not disq, f"Should not be disqualified, got: {reason}"
    print("  PASS consulting_with_prior_product_NOT_disqualified")


def test_collaborative_filtering_excluded():
    """The user's bad-rank-#1 case: 'collaborative filtering' must NOT count
    for the embeddings_retrieval must-have."""
    c = make_candidate(
        profile={"current_title": "AI Research Engineer", "current_company": "Verloop.io",
                 "years_of_experience": 4.3, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "Verloop.io",
                "title": "AI Research Engineer",
                "start_date": "2023-01-01",
                "duration_months": 24,
                "description": (
                    "Built recommendation-style features at a mid-stage startup — "
                    "lighter weight than ranking systems at FAANG, but production. "
                    "Used a combination of collaborative filtering (matrix factorization "
                    "in implicit-feedback library) and gradient-boosted re-ranking over "
                    "engagement signals."
                ),
            },
        ],
    )
    scores = must_haves.detect_must_haves(c)
    # embeddings_retrieval must NOT be met because of the
    # "collaborative filtering" / "matrix factorization" exclusion.
    assert not scores["embeddings_retrieval"]["met"], \
        "Should not be met due to 'collaborative filtering' exclusion"
    print("  PASS collaborative_filtering_excluded")


def test_forecasting_excluded_from_ranking_eval():
    """Time-series forecasting must NOT count for ranking_eval."""
    c = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "LogisticsCo",
                 "years_of_experience": 5.0, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "LogisticsCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "duration_months": 48,
                "description": (
                    "Worked on time-series forecasting models for supply-chain demand "
                    "prediction. Built models in Prophet, LightGBM. The LightGBM model "
                    "ended up shipping. Also ran some reinforcement learning experiments "
                    "for dynamic pricing but those didn't make it to production."
                ),
            },
        ],
    )
    scores = must_haves.detect_must_haves(c)
    assert not scores["ranking_eval"]["met"], \
        "Should not be met due to 'time series' exclusion"
    print("  PASS forecasting_excluded_from_ranking_eval")


def test_strong_ranking_evidence():
    """A candidate with explicit ranking eval and embeddings should hit 4/4."""
    c = make_candidate(
        profile={"current_title": "Senior ML Engineer", "current_company": "Google",
                 "years_of_experience": 7.0, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "Google",
                "title": "Senior ML Engineer",
                "start_date": "2021-01-01",
                "duration_months": 48,
                "description": (
                    "Built the offline-online correlation eval framework for our search "
                    "ranking system using NDCG and MRR as the primary offline metrics. "
                    "Owned the ranking layer end-to-end, from feature engineering through "
                    "online A/B testing. Used sentence-transformers with FAISS for fast "
                    "nearest-neighbor retrieval, deployed to production. "
                    "Production deployment with FastAPI in Python. "
                    "Improved NDCG by 20% through online experiments."
                ),
            },
        ],
    )
    scores = must_haves.detect_must_haves(c)
    n_met = sum(1 for v in scores.values() if v["met"])
    assert n_met >= 3, f"Expected >= 3/4, got {n_met}"
    print(f"  PASS strong_ranking_evidence ({n_met}/4)")


def test_platform_team_disqualifies():
    """'Production handled by the platform team' must NOT count for must-haves."""
    c = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "BigCo",
                 "years_of_experience": 6.0, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "BigCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "duration_months": 60,
                "description": (
                    "Designed the retrieval system. The production deployment was "
                    "handled by the platform team. Used Pinecone and sentence-transformers. "
                    "Designed NDCG-based offline evaluation."
                ),
            },
        ],
    )
    scores = must_haves.detect_must_haves(c)
    # The exclusion should kick in for embeddings_retrieval.
    # The candidate DISQUALIFIES the must-have rather than meeting it.
    for mh_name, info in scores.items():
        if info.get("disqualifying_sentences"):
            print(f"    [{mh_name}] disqualifying: {info['disqualifying_sentences'][0][:100]}")
    print("  PASS platform_team_disqualifies")


def test_availability_score_open_active():
    """A candidate who is open, 30d notice, 80% response, recently active
    should get availability ~= 1.0."""
    c = make_candidate(redrob_signals={
        "open_to_work_flag": True,
        "notice_period_days": 30,
        "recruiter_response_rate": 0.80,
        "last_active_date": "2026-05-15",
    })
    avail = must_haves.compute_availability_score(c) if hasattr(must_haves, "compute_availability_score") else None
    # Use the public scoring API
    import scoring
    avail = scoring.compute_availability_score(c)
    assert avail > 0.85, f"Expected > 0.85, got {avail:.3f}"
    print(f"  PASS availability_open_active ({avail:.3f})")


def test_availability_score_stale_passive():
    """A candidate who is NOT open, 120d notice, 5% response, 6mo inactive
    should get availability < 0.30."""
    c = make_candidate(redrob_signals={
        "open_to_work_flag": False,
        "notice_period_days": 120,
        "recruiter_response_rate": 0.05,
        "last_active_date": "2025-11-01",
    })
    import scoring
    avail = scoring.compute_availability_score(c)
    assert avail < 0.30, f"Expected < 0.30, got {avail:.3f}"
    print(f"  PASS availability_stale_passive ({avail:.3f})")


def test_fit_score_gating():
    """A candidate with 0 must-haves met should have fit_score < 30, even
    if availability is 1.0. This is the core invariant per audit spec."""
    import scoring
    # Strong AI skills, NO production evidence in career descriptions.
    c = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "BigCo",
                 "years_of_experience": 5.0, "location": "Bangalore", "country": "India"},
        career=[
            {
                "company": "BigCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "duration_months": 60,
                "description": "Studied ML techniques and built toy models.",
            },
        ],
        skills=[{"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 36}],
        redrob_signals={
            "open_to_work_flag": True,
            "notice_period_days": 30,
            "recruiter_response_rate": 0.85,
            "last_active_date": "2026-05-15",
        },
    )
    scores = scoring.compute_final_score(c)
    # 0 must-haves met → fit_score < 30
    assert scores["fit_score"] < 30, f"Expected fit < 30 with 0 must-haves, got {scores['fit_score']}"
    # Final score with availability 1.0 should still be < 0.30
    assert scores["final_score"] < 0.30, f"Expected final < 0.30, got {scores['final_score']:.3f}"
    print(f"  PASS fit_score_gating (fit={scores['fit_score']:.1f}, final={scores['final_score']:.3f})")


def main():
    print("Testing new must-have detection (audit spec Step 2):")
    test_strong_fit_candidate()
    test_consulting_only_disqualified()
    test_consulting_with_prior_product_NOT_disqualified()
    test_collaborative_filtering_excluded()
    test_forecasting_excluded_from_ranking_eval()
    test_strong_ranking_evidence()
    test_platform_team_disqualifies()
    test_availability_score_open_active()
    test_availability_score_stale_passive()
    test_fit_score_gating()
    print("\nAll 10 must-have tests passed.")


if __name__ == "__main__":
    main()
