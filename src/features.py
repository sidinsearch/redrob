"""features.py — Extract every signal used by scoring and reasoning.

The Feature dataclass is the single per-candidate record passed from
extract_features() to scoring. All downstream modules (trap_detector,
scoring, reasoning) consume this — they should NOT re-read the raw
candidate dict. Centralizing extraction makes the pipeline auditable:
"what signal could possibly affect this score?" → grep for it here.

ponytail: this file is large (~400 lines) but every function is a pure
extractor with no shared mutable state. Easy to unit-test in isolation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

import config
import utils


# ----------------------------------------------------------------------------
# Data class — the single per-candidate record
# ----------------------------------------------------------------------------

@dataclass
class Features:
    """All extracted features for one candidate.

    Fields are populated by extract_features() in a single pass over the
    candidate dict. They are read by trap_detector, scoring, and reasoning.
    """
    # Identity
    candidate_id: str

    # Profile
    current_title: str
    current_company: str
    current_industry: str
    years_of_experience: float
    country: str
    location: str

    # Title relevance
    title_score: float = 0.0

    # Experience fit
    experience_fit: float = 0.0

    # Product / consulting split
    is_consulting_current: bool = False
    has_product_company_history: bool = False
    career_product_ratio: float = 0.0  # 0.0–1.0
    product_exp_score: float = 0.0

    # Skills
    ai_skill_count: int = 0
    has_foundational_ml: bool = False
    has_production_ai_tools: bool = False
    has_ranking_skills: bool = False
    has_llm_buzzwords_only: bool = False
    max_skill_proficiency: str = "beginner"
    total_skill_count: int = 0
    ai_skills_depth: float = 0.0

    # Career history — aggregated
    career_ai_keyword_hits: int = 0
    has_ai_title_in_history: bool = False
    career_history: list = field(default_factory=list)

    # Education
    best_education_tier: str = "unknown"
    best_education_field: str = ""
    education_score: float = 0.0

    # Behavioral signals (raw)
    profile_completeness: float = 0.0
    open_to_work: bool = False
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False

    # Derived
    behavioral_score: float = 0.0
    location_score: float = 0.0
    availability_score: float = 0.0

    # Career — title chasing
    num_jobs_8yr: int = 0
    avg_tenure_months_8yr: float = 0.0
    is_title_chaser: bool = False

    # Trap flags
    template_summary_match: bool = False

    # Honeypot flags
    career_timeline_anomaly: bool = False
    expert_with_zero_duration: bool = False
    title_skills_history_mismatch: bool = False

    # Pre-built blob for fast matching (career_relevance uses this)
    search_blob: str = ""

    # For evidence-based reasoning — matched JD must-haves
    matched_must_haves: List[str] = field(default_factory=list)

    # The candidate_id of the matched sample-rejection phrase
    has_rejection_phrase: bool = False


# ----------------------------------------------------------------------------
# Title scoring
# ----------------------------------------------------------------------------

def _score_title(title: str) -> float:
    if not title:
        return config.DEFAULT_TITLE_SCORE
    t = title.lower()
    best = config.DEFAULT_TITLE_SCORE
    for score, needles in config.TITLE_CATEGORIES:
        for needle in needles:
            if needle in t:
                if score > best:
                    best = score
                break
    return best


# ----------------------------------------------------------------------------
# Product vs consulting
# ----------------------------------------------------------------------------

def _classify_company(company: str) -> str:
    """Return 'consulting', 'product', or 'unknown' for a single company name."""
    if not company:
        return "unknown"
    c = company.lower().strip()
    if c in config.CONSULTING_COMPANIES:
        return "consulting"
    if c in config.PRODUCT_COMPANY_HINTS:
        return "product"
    # Heuristic: contains a known consulting keyword
    for consulting_name in config.CONSULTING_COMPANIES:
        if consulting_name in c:
            return "consulting"
    return "unknown"


def _compute_product_exp(
    current_company: str,
    career_history: list,
) -> tuple:
    """Compute (product_exp_score, career_product_ratio, has_product_history, is_consulting_current)."""
    # 1) Current company classification
    current_class = _classify_company(current_company)
    is_consulting_current = current_class == "consulting"

    # 2) Career history classification
    if not career_history:
        return config.PRODUCT_EXP_PURE_CONSULTING, 0.0, False, is_consulting_current

    classes = [_classify_company(c.get("company", "")) for c in career_history]
    product_count = sum(1 for cls in classes if cls == "product")
    consulting_count = sum(1 for cls in classes if cls == "consulting")
    has_product = product_count > 0
    has_consulting = consulting_count > 0

    # If we have at least one product company in history, mark as product history
    if has_product:
        # Mixed or all product
        if has_consulting and product_count < consulting_count:
            return config.PRODUCT_EXP_MIXED, product_count / len(classes), True, is_consulting_current
        return config.PRODUCT_EXP_ALL_PRODUCT, product_count / len(classes), True, is_consulting_current
    elif has_consulting:
        # All consulting
        return config.PRODUCT_EXP_PURE_CONSULTING, 0.0, False, is_consulting_current
    else:
        # All unknown — treat as "no consulting penalty"
        return config.PRODUCT_EXP_MIXED, 0.5, False, is_consulting_current


# ----------------------------------------------------------------------------
# Skills analysis
# ----------------------------------------------------------------------------

def _analyze_skills(skills: list) -> dict:
    """Returns a dict of skill-derived features.

    Note: skill 'name' might be a list (in some malformed candidates) or
    a string. We coerce to string.
    """
    out = {
        "total_skill_count": len(skills),
        "ai_skill_count": 0,
        "has_foundational_ml": False,
        "has_production_ai_tools": False,
        "has_ranking_skills": False,
        "has_llm_buzzwords_only": False,
        "max_skill_proficiency": "beginner",
    }
    if not skills:
        return out

    # Proficiency levels ordered
    PROF_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    max_prof = 0

    foundational_hits = 0
    production_hits = 0
    ranking_hits = 0
    buzzword_hits = 0
    total_ai_hits = 0  # all AI-related including buzzwords

    for s in skills:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").lower()
        prof = (s.get("proficiency") or "beginner").lower()
        max_prof = max(max_prof, PROF_ORDER.get(prof, 0))

        # Substring check across categories
        is_ai = False
        for needle in config.FOUNDATIONAL_ML_SKILLS:
            if needle in name:
                foundational_hits += 1
                is_ai = True
                break
        for needle in config.PRODUCTION_AI_TOOLS:
            if needle in name:
                production_hits += 1
                is_ai = True
                break
        for needle in config.RANKING_SKILLS:
            if needle in name:
                ranking_hits += 1
                is_ai = True
                break
        for needle in config.LLM_BUZZWORD_SKILLS:
            if needle in name:
                buzzword_hits += 1
                is_ai = True
                break

        if is_ai:
            total_ai_hits += 1

    out["ai_skill_count"] = total_ai_hits
    out["has_foundational_ml"] = foundational_hits > 0
    out["has_production_ai_tools"] = production_hits > 0
    out["has_ranking_skills"] = ranking_hits > 0
    out["has_llm_buzzwords_only"] = (
        not out["has_foundational_ml"]
        and not out["has_ranking_skills"]
        and buzzword_hits >= config.KEYWORD_STUFFER_MIN_BUZZWORDS
    )
    out["max_skill_proficiency"] = next(
        k for k, v in PROF_ORDER.items() if v == max_prof
    )

    # AI skills depth — composite
    depth = 0.0
    if out["has_foundational_ml"]:
        depth += 0.4
    if out["has_production_ai_tools"]:
        depth += 0.2
    if out["has_ranking_skills"]:
        depth += 0.2
    # Bonus for high proficiency on AI skills
    ai_expert_count = 0
    for s in skills:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").lower()
        prof = (s.get("proficiency") or "beginner").lower()
        if prof in ("advanced", "expert") and any(n in name for n in config.FOUNDATIONAL_ML_SKILLS + config.RANKING_SKILLS):
            ai_expert_count += 1
    depth += min(0.1, ai_expert_count * 0.025)
    # Endorsement bonus
    endorsement_total = 0
    for s in skills:
        if isinstance(s, dict):
            e = s.get("endorsements", 0)
            if isinstance(e, (int, float)):
                endorsement_total += int(e)
    depth += min(0.1, endorsement_total / 500.0)

    # Normalize against total skill count to penalize inflation
    if out["total_skill_count"] > config.SKILL_INFLATION_THRESHOLD:
        depth *= 0.7  # heavy inflation penalty
    elif out["total_skill_count"] > 12:
        depth *= 0.85

    out["ai_skills_depth"] = utils.clip01(depth)
    return out


# ----------------------------------------------------------------------------
# Career analysis
# ----------------------------------------------------------------------------

def _analyze_career(career_history: list, search_blob: str) -> dict:
    """Extract career-derived features."""
    out = {
        "career_ai_keyword_hits": 0,
        "has_ai_title_in_history": False,
        "num_jobs_8yr": 0,
        "avg_tenure_months_8yr": 0.0,
        "is_title_chaser": False,
        "matched_must_haves": [],
    }

    if not career_history:
        return out

    # AI keyword hits across all career descriptions
    hits = 0
    for c in career_history:
        desc = (c.get("description") or "").lower()
        title = (c.get("title") or "").lower()
        for kw in config.CAREER_AI_KEYWORDS:
            if kw in desc or kw in title:
                hits += 1
                break  # one hit per role
    out["career_ai_keyword_hits"] = hits

    # AI title in history
    AI_TITLE_PATTERNS = [
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "applied scientist", "research engineer", "nlp", "computer vision",
        "deep learning", "recommendation", "ranking", "search engineer",
        "retrieval",
    ]
    for c in career_history:
        title = (c.get("title") or "").lower()
        for pat in AI_TITLE_PATTERNS:
            if pat in title:
                out["has_ai_title_in_history"] = True
                break
        if out["has_ai_title_in_history"]:
            break

    # Title-chaser: 4+ jobs in last 8 years with avg tenure <18 months
    now = date.today()
    eight_years_ago = date(now.year - config.TITLE_CHASER_LOOKBACK_YEARS, now.month, now.day)
    recent_jobs = []
    for c in career_history:
        start_str = c.get("start_date", "")
        try:
            start = date.fromisoformat(start_str) if start_str else None
        except (ValueError, TypeError):
            start = None
        if start and start >= eight_years_ago:
            recent_jobs.append(c)

    out["num_jobs_8yr"] = len(recent_jobs)
    if len(recent_jobs) >= config.TITLE_CHASER_MIN_JOBS:
        total_months = sum(int(c.get("duration_months") or 0) for c in recent_jobs)
        avg = total_months / len(recent_jobs)
        out["avg_tenure_months_8yr"] = avg
        out["is_title_chaser"] = avg < config.TITLE_CHASER_AVG_TENURE_MONTHS

    # JD must-have match — for evidence-based reasoning
    for category, patterns in config.JD_MUST_HAVE_PATTERNS.items():
        if any(p in search_blob for p in patterns):
            out["matched_must_haves"].append(category)

    return out


# ----------------------------------------------------------------------------
# Education
# ----------------------------------------------------------------------------

def _analyze_education(education: list) -> dict:
    out = {
        "best_education_tier": "unknown",
        "best_education_field": "",
        "education_score": config.EDUCATION_DEFAULT_SCORE,
    }
    if not education:
        return out
    # Find best tier
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    best_tier = "unknown"
    best_field = ""
    for e in education:
        if not isinstance(e, dict):
            continue
        tier = e.get("tier", "unknown")
        if tier_rank.get(tier, 0) > tier_rank.get(best_tier, 0):
            best_tier = tier
            best_field = e.get("field_of_study", "") or ""
    out["best_education_tier"] = best_tier
    out["best_education_field"] = best_field
    out["education_score"] = config.EDUCATION_TIER_SCORES.get(best_tier, config.EDUCATION_DEFAULT_SCORE)
    return out


# ----------------------------------------------------------------------------
# Behavioral score
# ----------------------------------------------------------------------------

def _compute_behavioral_score(rs: dict) -> float:
    """Normalize and combine behavioral signals into [0, 1]."""
    # Use ceilings tuned from the data (search_appearance ~50 max in our
    # sample, saved_by_recruiters ~20, endorsements ~100, connections ~1000,
    # response_time ~ 200h ceiling).
    score = 0.0
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["search_appearance_30d"] * utils.log_normalize(
        rs.get("search_appearance_30d", 0), ceiling=500.0
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["saved_by_recruiters_30d"] * utils.log_normalize(
        rs.get("saved_by_recruiters_30d", 0), ceiling=30.0
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["endorsements_received"] * utils.log_normalize(
        rs.get("endorsements_received", 0), ceiling=200.0
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["connection_count"] * utils.log_normalize(
        rs.get("connection_count", 0), ceiling=1000.0
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["recruiter_response_rate"] * utils.clip01(
        rs.get("recruiter_response_rate", 0.0)
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["avg_response_time_hours"] * utils.inverse_normalize(
        rs.get("avg_response_time_hours", 0.0), ceiling=200.0
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["interview_completion_rate"] * utils.clip01(
        rs.get("interview_completion_rate", 0.0)
    )
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["open_to_work_flag"] * (
        1.0 if rs.get("open_to_work_flag", False) else 0.0
    )
    gh = rs.get("github_activity_score", -1.0)
    if isinstance(gh, (int, float)) and gh > 0:
        score += config.BEHAVIORAL_SIGNAL_WEIGHTS["github_activity_score"] * utils.clip01(gh / 50.0)
    pc = rs.get("profile_completeness_score", 0.0)
    score += config.BEHAVIORAL_SIGNAL_WEIGHTS["profile_completeness_score"] * utils.clip01(pc / 100.0)
    return utils.clip01(score)


# ----------------------------------------------------------------------------
# Location
# ----------------------------------------------------------------------------

def _compute_location_score(country: str, location: str) -> float:
    base = config.LOCATION_SCORES.get(utils.lc(country).strip(), config.LOCATION_DEFAULT_SCORE)
    if utils.lc(country).strip() == "india":
        loc = utils.lc(location).strip()
        for city, boost in config.INDIAN_CITY_BOOST.items():
            if city in loc:
                base = min(config.INDIAN_CITY_BOOST_MAX, base + boost)
                break
    return base


# ----------------------------------------------------------------------------
# Honeypot detection
# ----------------------------------------------------------------------------

def _detect_honeypot_signals(
    profile: dict,
    career_history: list,
    skills: list,
    features: Features,
) -> None:
    """Mutates features in-place to set honeypot flags.

    Honeypots are impossibly wrong profiles. We check three patterns:

    1. Career timeline anomaly: YoE > (current_company_start_year - earliest_career_start) + buffer.
    2. Expert with zero duration: "expert" in 5+ skills but 0 months of use.
    3. Title-skills-history mismatch: "Senior AI Engineer" with 0 AI skills AND no AI history.
    """
    # 1. Career timeline anomaly
    yoe = profile.get("years_of_experience", 0) or 0
    # Compute total career span
    if career_history:
        starts = []
        for c in career_history:
            s = c.get("start_date", "")
            try:
                starts.append(date.fromisoformat(s) if s else None)
            except (ValueError, TypeError):
                continue
        starts = [d for d in starts if d is not None]
        if starts:
            earliest = min(starts)
            span_years = (date.today() - earliest).days / 365.25
            # If YoE claims more than (career span + 5yr), it's impossible
            if yoe > span_years + config.HONEYPOT_YOE_BUFFER_YEARS:
                features.career_timeline_anomaly = True

    # 2. Expert with zero duration
    expert_count = 0
    for s in skills:
        if not isinstance(s, dict):
            continue
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months", 0) or 0
        if prof == "expert" and (not isinstance(dur, (int, float)) or dur == 0):
            expert_count += 1
    if expert_count >= config.EXPERT_SKILL_FAKE_THRESHOLD:
        features.expert_with_zero_duration = True

    # 3. Title-skills-history mismatch
    title_lc = (profile.get("current_title", "") or "").lower()
    has_ai_title = any(
        kw in title_lc for kw in [
            "ai engineer", "ml engineer", "data scientist", "nlp",
            "machine learning", "deep learning", "research engineer",
            "applied scientist", "ai/ml",
        ]
    )
    if has_ai_title:
        # Check if the profile has any real AI signal
        if (
            not features.has_foundational_ml
            and not features.has_ranking_skills
            and not features.has_ai_title_in_history
            and features.ai_skill_count == 0
        ):
            features.title_skills_history_mismatch = True


# ----------------------------------------------------------------------------
# Main extraction
# ----------------------------------------------------------------------------

def extract_features(candidate: dict) -> Features:
    """Extract all features for a single candidate. Returns a Features dataclass."""
    profile = candidate.get("profile", {}) or {}
    career_history = utils.list_field(candidate, "career_history")
    skills = utils.list_field(candidate, "skills")
    education = utils.list_field(candidate, "education")
    rs = candidate.get("redrob_signals", {}) or {}

    search_blob = utils.build_search_blob(candidate)

    # Product / consulting
    product_exp_score, product_ratio, has_product_history, is_consulting_current = _compute_product_exp(
        profile.get("current_company", ""), career_history
    )

    # Skills
    skill_features = _analyze_skills(skills)

    # Career
    career_features = _analyze_career(career_history, search_blob)

    # Education
    edu_features = _analyze_education(education)

    # Behavioral
    behavioral = _compute_behavioral_score(rs)

    # Location
    location = _compute_location_score(profile.get("country", ""), profile.get("location", ""))

    # Notice period
    notice = int(rs.get("notice_period_days") or 0)
    availability = config.notice_period_score(notice)

    # Template summary trap
    summary = utils.lc(profile.get("summary", ""))
    template_match = config.TEMPLATE_SUMMARY_PHRASE.lower() in summary

    # Build features
    f = Features(
        candidate_id=candidate.get("candidate_id", ""),
        current_title=profile.get("current_title", "") or "",
        current_company=profile.get("current_company", "") or "",
        current_industry=profile.get("current_industry", "") or "",
        years_of_experience=float(profile.get("years_of_experience") or 0),
        country=profile.get("country", "") or "",
        location=profile.get("location", "") or "",
        title_score=_score_title(profile.get("current_title", "") or ""),
        experience_fit=config.experience_fit(float(profile.get("years_of_experience") or 0)),
        is_consulting_current=is_consulting_current,
        has_product_company_history=has_product_history,
        career_product_ratio=product_ratio,
        product_exp_score=product_exp_score,
        total_skill_count=skill_features["total_skill_count"],
        ai_skill_count=skill_features["ai_skill_count"],
        has_foundational_ml=skill_features["has_foundational_ml"],
        has_production_ai_tools=skill_features["has_production_ai_tools"],
        has_ranking_skills=skill_features["has_ranking_skills"],
        has_llm_buzzwords_only=skill_features["has_llm_buzzwords_only"],
        max_skill_proficiency=skill_features["max_skill_proficiency"],
        ai_skills_depth=skill_features["ai_skills_depth"],
        career_ai_keyword_hits=career_features["career_ai_keyword_hits"],
        has_ai_title_in_history=career_features["has_ai_title_in_history"],
        career_history=list(career_history),
        best_education_tier=edu_features["best_education_tier"],
        best_education_field=edu_features["best_education_field"],
        education_score=edu_features["education_score"],
        profile_completeness=float(rs.get("profile_completeness_score") or 0),
        open_to_work=bool(rs.get("open_to_work_flag", False)),
        recruiter_response_rate=float(rs.get("recruiter_response_rate") or 0),
        avg_response_time_hours=float(rs.get("avg_response_time_hours") or 0),
        connection_count=int(rs.get("connection_count") or 0),
        endorsements_received=int(rs.get("endorsements_received") or 0),
        notice_period_days=notice,
        github_activity_score=float(rs.get("github_activity_score") or -1.0),
        search_appearance_30d=int(rs.get("search_appearance_30d") or 0),
        saved_by_recruiters_30d=int(rs.get("saved_by_recruiters_30d") or 0),
        interview_completion_rate=float(rs.get("interview_completion_rate") or 0),
        offer_acceptance_rate=float(rs.get("offer_acceptance_rate") or -1.0),
        verified_email=bool(rs.get("verified_email", False)),
        verified_phone=bool(rs.get("verified_phone", False)),
        behavioral_score=behavioral,
        location_score=location,
        availability_score=availability,
        num_jobs_8yr=career_features["num_jobs_8yr"],
        avg_tenure_months_8yr=career_features["avg_tenure_months_8yr"],
        is_title_chaser=career_features["is_title_chaser"],
        template_summary_match=template_match,
        search_blob=search_blob,
        matched_must_haves=career_features["matched_must_haves"],
    )

    # Honeypot detection (mutates f)
    _detect_honeypot_signals(profile, career_history, skills, f)

    return f
