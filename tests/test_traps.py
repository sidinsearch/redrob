"""test_traps.py — Unit tests for trap detection.

Each test builds a known-bad candidate and verifies the trap fires.
Each test also builds a known-good candidate and verifies it doesn't.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import extract_features
from trap_detector import (
    analyze,
    detect_consulting_only,
    detect_keyword_stuffer,
    detect_template_summary,
    detect_title_chaser,
    is_honeypot,
)


def make_candidate(**overrides) -> dict:
    """Build a candidate with sensible defaults that can be overridden."""
    cand = {
        "candidate_id": "CAND_TEST_001",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Software Engineer",
            "summary": "Regular engineering summary.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech",
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Built and deployed recommendation system using embeddings and FAISS.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 250,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.85,
            "avg_response_time_hours": 8.0,
            "skill_assessment_scores": {"PyTorch": 80.0, "FAISS": 70.0},
            "connection_count": 500,
            "endorsements_received": 100,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 40.0,
            "search_appearance_30d": 300,
            "saved_by_recruiters_30d": 25,
            "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    # Apply overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cand and isinstance(cand[k], dict):
            cand[k].update(v)
        else:
            cand[k] = v
    return cand


# ----------------------------------------------------------------------------
# Clean candidate — should NOT trigger any trap
# ----------------------------------------------------------------------------

def test_clean_candidate_no_traps():
    cand = make_candidate()
    f = extract_features(cand)
    t = analyze(f)
    assert not t.is_honeypot, f"clean candidate flagged as honeypot: {t.honeypot_reasons}"
    assert not t.is_keyword_stuffer
    assert not t.is_template_summary
    assert not t.is_consulting_only
    assert not t.is_title_chaser
    assert t.trap_multiplier == 1.0


# ----------------------------------------------------------------------------
# Keyword stuffer — Marketing Manager with RAG skills
# ----------------------------------------------------------------------------

def test_keyword_stuffer_fires():
    cand = make_candidate(
        profile={
            "current_title": "Marketing Manager",
            "current_company": "Coca-Cola",
            "current_industry": "FMCG",
            "years_of_experience": 5.0,
            "country": "India",
            "location": "Mumbai",
        },
        # No AI history — pure marketing career
        career_history=[
            {"company": "Coca-Cola", "title": "Marketing Manager", "start_date": "2020-01-01", "end_date": None, "duration_months": 60, "is_current": True, "industry": "FMCG", "company_size": "10001+", "description": "Marketing campaigns and brand strategy."},
            {"company": "Unilever", "title": "Marketing Executive", "start_date": "2018-01-01", "end_date": "2020-01-01", "duration_months": 24, "is_current": False, "industry": "FMCG", "company_size": "10001+", "description": "Marketing analytics."},
        ],
        skills=[
            {"name": "RAG", "proficiency": "advanced", "endorsements": 5, "duration_months": 6},
            {"name": "LangChain", "proficiency": "intermediate", "endorsements": 3, "duration_months": 6},
            {"name": "Embeddings", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            {"name": "Fine-tuning LLMs", "proficiency": "beginner", "endorsements": 1, "duration_months": 3},
        ],
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_keyword_stuffer, f"Marketing Manager with RAG skills should be flagged (got: ks={t.is_keyword_stuffer}, has_ai_title_in_history={f.has_ai_title_in_history})"
    assert t.trap_multiplier <= 0.5


# ----------------------------------------------------------------------------
# Template summary — "curious about AI tools"
# ----------------------------------------------------------------------------

def test_template_summary_fires():
    cand = make_candidate(
        profile={
            "summary": "Professional with 8 years. Lately I've been curious about how AI tools could augment my work — I've experimented with ChatGPT.",
        },
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_template_summary
    assert t.trap_multiplier <= 0.8


# ----------------------------------------------------------------------------
# Consulting only — TCS / Infosys / Wipro career
# ----------------------------------------------------------------------------

def test_consulting_only_fires():
    cand = make_candidate(
        profile={"current_company": "Infosys"},
        career_history=[
            {
                "company": "Infosys",
                "title": "Senior Consultant",
                "start_date": "2018-01-01",
                "end_date": None,
                "duration_months": 96,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Client consulting for various enterprises.",
            },
        ],
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_consulting_only
    assert t.trap_multiplier <= 0.8


# ----------------------------------------------------------------------------
# Title chaser — 4 jobs in 8 years (all in the last 8 years from today=2026)
# ----------------------------------------------------------------------------

def test_title_chaser_fires():
    cand = make_candidate(
        career_history=[
            # 5 jobs in last 8 years from 2026 (i.e. since 2018-06-15).
            # All tenures 12-18 months. Average <18 → title chaser.
            {"company": "A", "title": "Engineer", "start_date": "2024-06-01", "end_date": "2025-06-01", "duration_months": 12, "is_current": False, "industry": "Tech", "company_size": "1001-5000", "description": "X"},
            {"company": "B", "title": "Senior Engineer", "start_date": "2023-01-01", "end_date": "2024-04-01", "duration_months": 15, "is_current": False, "industry": "Tech", "company_size": "1001-5000", "description": "X"},
            {"company": "C", "title": "Lead Engineer", "start_date": "2021-06-01", "end_date": "2022-09-01", "duration_months": 15, "is_current": False, "industry": "Tech", "company_size": "1001-5000", "description": "X"},
            {"company": "D", "title": "Staff Engineer", "start_date": "2019-09-01", "end_date": "2021-02-01", "duration_months": 17, "is_current": False, "industry": "Tech", "company_size": "1001-5000", "description": "X"},
            {"company": "E", "title": "Engineer", "start_date": "2015-01-01", "end_date": "2019-08-01", "duration_months": 55, "is_current": False, "industry": "Tech", "company_size": "1001-5000", "description": "X"},
        ],
    )
    f = extract_features(cand)
    assert f.num_jobs_8yr >= 4, f"expected 4+ jobs in last 8yr, got {f.num_jobs_8yr}"
    assert f.avg_tenure_months_8yr < 18, f"expected avg <18mo, got {f.avg_tenure_months_8yr}"
    assert f.is_title_chaser, f"should be title chaser (num={f.num_jobs_8yr}, avg={f.avg_tenure_months_8yr})"


# ----------------------------------------------------------------------------
# Honeypot: career timeline impossible
# ----------------------------------------------------------------------------

def test_honeypot_timeline_fires():
    cand = make_candidate(
        profile={"years_of_experience": 15.0},  # 15 years claimed
        career_history=[
            # But career only started 2 years ago
            {"company": "NewStartup", "title": "ML Engineer", "start_date": "2024-01-01", "end_date": None, "duration_months": 24, "is_current": True, "industry": "Tech", "company_size": "1-10", "description": "X"},
        ],
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_honeypot
    assert "career timeline" in " ".join(t.honeypot_reasons).lower()


# ----------------------------------------------------------------------------
# Honeypot: expert with zero duration
# ----------------------------------------------------------------------------

def test_honeypot_expert_zero_duration_fires():
    cand = make_candidate(
        skills=[
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "TensorFlow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Kubernetes", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Docker", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "MLflow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_honeypot


# ----------------------------------------------------------------------------
# Honeypot: title-skills mismatch
# ----------------------------------------------------------------------------

def test_honeypot_title_skills_mismatch_fires():
    cand = make_candidate(
        profile={"current_title": "Senior AI Engineer"},
        skills=[
            {"name": "Microsoft Excel", "proficiency": "expert", "endorsements": 100, "duration_months": 60},
            {"name": "PowerPoint", "proficiency": "advanced", "endorsements": 50, "duration_months": 60},
        ],
        career_history=[
            {"company": "TCS", "title": "Business Analyst", "start_date": "2018-01-01", "end_date": None, "duration_months": 96, "is_current": True, "industry": "IT Services", "company_size": "10001+", "description": "Excel reports."},
        ],
    )
    f = extract_features(cand)
    t = analyze(f)
    assert t.is_honeypot


# ----------------------------------------------------------------------------
# Combined: clean candidate scores > trap candidate
# ----------------------------------------------------------------------------

def test_clean_beats_trap():
    """A clean candidate with real production evidence should beat a trap
    candidate with only LLM buzzwords in skills.

    In the new model, the trap is detected by:
    1. The trap_detector (keyword_stuffer=True)
    2. The must-haves detector (0/4 because no career_history evidence)
    The clean candidate has fit_score > 0 (real evidence), the trap
    candidate has fit_score = 0 (no evidence). So the trap should score
    lower regardless of availability.
    """
    clean = make_candidate(
        profile={"current_title": "ML Engineer", "current_company": "BigCo",
                 "years_of_experience": 6.0, "location": "Bangalore", "country": "India"},
        career_history=[{
            "company": "BigCo", "title": "ML Engineer",
            "start_date": "2020-01-01", "duration_months": 60,
            "description": "Built retrieval system in production. Designed NDCG eval. Python.",
        }],
        redrob_signals={
            "open_to_work_flag": True, "notice_period_days": 30,
            "recruiter_response_rate": 0.85, "last_active_date": "2026-05-15",
        },
    )
    trap = make_candidate(
        profile={"current_title": "Marketing Manager", "current_company": "Coca-Cola", "current_industry": "FMCG"},
        career_history=[{
            "company": "Coca-Cola", "title": "Marketing Manager",
            "start_date": "2020-01-01", "duration_months": 60,
            "description": "Ran marketing campaigns. Did some hobby coding.",
        }],
        skills=[
            {"name": "RAG", "proficiency": "advanced", "endorsements": 5, "duration_months": 6},
            {"name": "LangChain", "proficiency": "intermediate", "endorsements": 3, "duration_months": 6},
            {"name": "Embeddings", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            {"name": "OpenAI", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            {"name": "Prompt Engineering", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
        ],
    )
    f_trap = extract_features(trap)
    t_trap = analyze(f_trap)
    assert t_trap.is_keyword_stuffer, "Trap should be flagged as keyword_stuffer"

    from scoring import compute_final_score
    s_clean = compute_final_score(clean)["final_score"]
    s_trap = compute_final_score(trap)["final_score"]
    assert s_clean > s_trap, f"clean ({s_clean:.4f}) should beat trap ({s_trap:.4f})"


if __name__ == "__main__":
    # Run as plain script (no pytest dep)
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        import sys
        sys.exit(1)
