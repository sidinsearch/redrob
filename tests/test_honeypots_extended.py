"""test_honeypots_extended.py — Test all 14 honeypot detectors with synthetic profiles.

Each test builds a minimal candidate that should trigger exactly one detector,
runs extract_features + analyze, and asserts the right flag is set.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features
import trap_detector


def _make_candidate(profile=None, career=None, skills=None, education=None):
    """Build a minimal candidate dict with sane defaults."""
    return {
        "candidate_id": "CAND_TEST",
        "profile": profile or {},
        "career_history": career or [],
        "skills": skills or [],
        "education": education or [],
        "redrob_signals": {},
    }


def _analyze(cand):
    f = features.extract_features(cand)
    t = trap_detector.analyze(f)
    return f, t


def test_synthetic_profile():
    """8+ AI skills, <4 YoE, <3 jobs, high completeness → synthetic."""
    cand = _make_candidate(
        profile={"years_of_experience": 3.0, "current_title": "AI Engineer"},
        career=[
            {"company": "StartupX", "title": "AI Engineer", "start_date": "2023-01-01",
             "end_date": None, "duration_months": 36, "description": "Built things."},
        ],
        skills=[
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
            {"name": "TensorFlow", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
            {"name": "LLMs", "proficiency": "advanced", "endorsements": 5, "duration_months": 18},
            {"name": "RAG", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "LangChain", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "Vector Search", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
        ],
    )
    cand["redrob_signals"]["profile_completeness_score"] = 90.0
    f, t = _analyze(cand)
    assert f.synthetic_profile, f"Expected synthetic_profile, got {f.synthetic_profile}"
    assert t.is_honeypot
    print("  PASS synthetic_profile")


def test_technology_age():
    """Career started 2018, claims 8 years of LangChain (released 2022)."""
    cand = _make_candidate(
        profile={"years_of_experience": 8.0, "current_title": "ML Engineer"},
        career=[
            {"company": "BigCo", "title": "ML Engineer", "start_date": "2018-01-01",
             "end_date": None, "duration_months": 96, "description": "Built ML."},
        ],
        skills=[
            # 8 years of LangChain (96 months) — would have started in 2018,
            # but LangChain was released in 2022.
            {"name": "LangChain", "proficiency": "expert", "endorsements": 10, "duration_months": 96},
        ],
    )
    f, t = _analyze(cand)
    assert f.technology_age_anomaly, f"Expected tech_age_anomaly, got {f.technology_age_anomaly}"
    assert t.is_honeypot
    print("  PASS technology_age_anomaly")


def test_title_responsibility_mismatch():
    """AI/ML title, but 0 AI/ML keywords in any job description."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Senior AI Engineer"},
        career=[
            {"company": "BigCo", "title": "Senior AI Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Built internal admin tools. Ran standups. Wrote Jira tickets."},
        ],
        skills=[
            {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
        ],
    )
    f, t = _analyze(cand)
    assert f.title_responsibility_mismatch, f"Expected title_responsibility_mismatch, got {f.title_responsibility_mismatch}"
    assert t.is_honeypot
    print("  PASS title_responsibility_mismatch")


def test_skill_experience_contradiction():
    """Advanced+ proficiency in 3+ skills with near-zero duration."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "ML Engineer"},
        career=[
            {"company": "BigCo", "title": "ML Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Built machine learning models for production search ranking."},
        ],
        skills=[
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
            {"name": "TensorFlow", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
            {"name": "XGBoost", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
        ],
    )
    f, t = _analyze(cand)
    assert f.skill_experience_contradiction, f"Expected skill_experience_contradiction, got {f.skill_experience_contradiction}"
    assert t.is_honeypot
    print("  PASS skill_experience_contradiction")


def test_career_progression_anomaly():
    """3+ level jumps in engineering track in 1 year."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Principal Engineer"},
        career=[
            {"company": "A", "title": "Junior Engineer", "start_date": "2020-01-01",
             "end_date": "2020-06-01", "duration_months": 5,
             "description": "Intern-level work."},
            {"company": "B", "title": "Engineer", "start_date": "2020-07-01",
             "end_date": "2020-09-01", "duration_months": 2,
             "description": "Mid-level."},
            {"company": "C", "title": "Principal Engineer", "start_date": "2020-10-01",
             "end_date": None, "duration_months": 12,
             "description": "Principal-level architecture."},
        ],
    )
    f, t = _analyze(cand)
    assert f.career_progression_anomaly, f"Expected career_progression_anomaly, got {f.career_progression_anomaly}"
    assert t.is_honeypot
    print("  PASS career_progression_anomaly")


def test_duration_integrity_violation():
    """Job with negative duration_months."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Engineer"},
        career=[
            {"company": "X", "title": "Engineer", "start_date": "2020-01-01",
             "end_date": "2024-01-01", "duration_months": -12,
             "description": "Worked on backend."},
        ],
        skills=[{"name": "Python", "proficiency": "intermediate", "duration_months": 36}],
    )
    f, t = _analyze(cand)
    assert f.duration_integrity_violation, f"Expected duration_integrity_violation, got {f.duration_integrity_violation}"
    assert t.is_honeypot
    print("  PASS duration_integrity_violation")


def test_education_timeline_anomaly():
    """Degree end_year in the future."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Engineer"},
        career=[{"company": "X", "title": "Engineer", "start_date": "2020-01-01",
                 "end_date": None, "duration_months": 60, "description": "Worked."}],
        education=[{"institution": "Test U", "degree": "B.Tech", "field_of_study": "CS",
                    "start_year": 2030, "end_year": 2034, "tier": "tier_2"}],
    )
    f, t = _analyze(cand)
    assert f.education_timeline_anomaly, f"Expected education_timeline_anomaly, got {f.education_timeline_anomaly}"
    assert t.is_honeypot
    print("  PASS education_timeline_anomaly")


def test_employment_overlap_anomaly():
    """4+ concurrent jobs at same time."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Engineer"},
        career=[
            {"company": f"C{i}", "title": "Engineer", "start_date": "2022-01-01",
             "end_date": "2024-01-01", "duration_months": 24,
             "description": "Worked at C{i}."} for i in range(5)
        ],
    )
    f, t = _analyze(cand)
    assert f.employment_overlap_anomaly, f"Expected employment_overlap_anomaly, got {f.employment_overlap_anomaly}"
    assert t.is_honeypot
    print("  PASS employment_overlap_anomaly")


def test_achievement_inflation():
    """3+ inflation keywords with no metrics."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Engineer",
                 "summary": "I'm a world-class, industry-leading, revolutionary engineer."},
        career=[
            {"company": "X", "title": "Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Worked on cutting-edge groundbreaking pioneering stuff."},
        ],
        skills=[{"name": "Python", "proficiency": "intermediate", "duration_months": 60}],
    )
    f, t = _analyze(cand)
    assert f.achievement_inflation, f"Expected achievement_inflation, got {f.achievement_inflation}"
    assert t.is_honeypot
    print("  PASS achievement_inflation")


def test_cross_field_inconsistency():
    """NLP title with 0 NLP in skills or career."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "NLP Engineer"},
        career=[
            {"company": "X", "title": "NLP Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Built internal tools with Python and Java."},
        ],
        skills=[
            {"name": "Python", "proficiency": "advanced", "duration_months": 60},
            {"name": "Java", "proficiency": "advanced", "duration_months": 60},
        ],
    )
    f, t = _analyze(cand)
    assert f.cross_field_inconsistency, f"Expected cross_field_inconsistency, got {f.cross_field_inconsistency}"
    assert t.is_honeypot
    print("  PASS cross_field_inconsistency")


def test_nlp_claim_without_evidence():
    """Summary says 'no NLP experience' but skills list NLP."""
    cand = _make_candidate(
        profile={"years_of_experience": 5.0, "current_title": "Data Engineer",
                 "summary": "I have no experience in NLP and want to transition into it."},
        career=[
            {"company": "X", "title": "Data Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Built data pipelines."},
        ],
        skills=[
            {"name": "NLP", "proficiency": "expert", "duration_months": 36},
            {"name": "Python", "proficiency": "advanced", "duration_months": 60},
        ],
    )
    f, t = _analyze(cand)
    assert f.nlp_claim_without_evidence, f"Expected nlp_claim_without_evidence, got {f.nlp_claim_without_evidence}"
    assert t.is_honeypot
    print("  PASS nlp_claim_without_evidence")


def test_pre_llm_signal_present():
    """Pre-2020 role with retrieval/ranking keywords → pre_llm_signal > 0."""
    cand = _make_candidate(
        profile={"years_of_experience": 10.0, "current_title": "Search Engineer"},
        career=[
            {"company": "SearchCo", "title": "Search Engineer", "start_date": "2017-01-01",
             "end_date": "2020-01-01", "duration_months": 36,
             "description": "Built Elasticsearch-based search ranking with BM25 and learning to rank."},
            {"company": "SearchCo", "title": "Search Engineer", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 60,
             "description": "Continued search relevance work, added vector search and FAISS."},
        ],
    )
    f, t = _analyze(cand)
    assert f.pre_llm_roles >= 1, f"Expected pre_llm_roles >= 1, got {f.pre_llm_roles}"
    assert f.pre_llm_signal > 0, f"Expected pre_llm_signal > 0, got {f.pre_llm_signal}"
    print(f"  PASS pre_llm_signal (roles={f.pre_llm_roles}, signal={f.pre_llm_signal:.3f})")


def test_no_false_positive():
    """A clean, plausible profile should NOT be flagged as honeypot."""
    cand = _make_candidate(
        profile={"years_of_experience": 7.0, "current_title": "ML Engineer",
                 "summary": "Experienced ML engineer focused on production ranking systems."},
        career=[
            {"company": "SearchCo", "title": "ML Engineer", "start_date": "2019-01-01",
             "end_date": "2022-01-01", "duration_months": 36,
             "description": "Built production search ranking and retrieval system with embeddings and FAISS."},
            {"company": "SearchCo", "title": "Senior ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 48,
             "description": "Continued retrieval and ranking work. Deployed to 50M users."},
        ],
        skills=[
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 20, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 48},
            {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 12, "duration_months": 48},
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 84},
        ],
    )
    f, t = _analyze(cand)
    assert not t.is_honeypot, f"False positive: clean profile flagged. Flags: {[k for k in ['career_timeline_anomaly','expert_with_zero_duration','title_skills_history_mismatch','employment_overlap_anomaly','duration_integrity_violation','title_responsibility_mismatch','skill_experience_contradiction','education_timeline_anomaly','career_progression_anomaly','achievement_inflation','technology_age_anomaly','synthetic_profile','cross_field_inconsistency','nlp_claim_without_evidence'] if getattr(f, k)]}"
    assert f.pre_llm_signal > 0, "Expected pre-LLM signal for pre-2020 role"
    print(f"  PASS no_false_positive (pre_llm_signal={f.pre_llm_signal:.3f})")


def main():
    print("Testing honeypot detectors:")
    test_synthetic_profile()
    test_technology_age()
    test_title_responsibility_mismatch()
    test_skill_experience_contradiction()
    test_career_progression_anomaly()
    test_duration_integrity_violation()
    test_education_timeline_anomaly()
    test_employment_overlap_anomaly()
    test_achievement_inflation()
    test_cross_field_inconsistency()
    test_nlp_claim_without_evidence()
    test_pre_llm_signal_present()
    test_no_false_positive()
    print("\nAll 13 detector tests passed.")


if __name__ == "__main__":
    main()
