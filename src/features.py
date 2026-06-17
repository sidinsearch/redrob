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

    # Honeypot flags (10 new detectors added)
    career_timeline_anomaly: bool = False
    expert_with_zero_duration: bool = False
    title_skills_history_mismatch: bool = False
    # New detectors
    duration_integrity_violation: bool = False
    skill_experience_contradiction: bool = False
    education_timeline_anomaly: bool = False
    career_progression_anomaly: bool = False
    achievement_inflation: bool = False
    technology_age_anomaly: bool = False
    synthetic_profile: bool = False
    cross_field_inconsistency: bool = False
    employment_overlap_anomaly: bool = False
    title_responsibility_mismatch: bool = False
    nlp_claim_without_evidence: bool = False

    # Pre-LLM signal (boost for pre-2020 retrieval/ranking/ML production work)
    pre_llm_signal: float = 0.0
    pre_llm_roles: int = 0

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
        "ai_skills_depth": 0.0,
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

    Honeypots are impossibly wrong profiles. We check 13 patterns:

    Existing (3):
    1. Career timeline anomaly: YoE > (career span + buffer).
    2. Expert with zero duration: "expert" in 5+ skills but 0 months of use.
    3. Title-skills-history mismatch: "Senior AI Engineer" with 0 AI skills AND no AI history.

    New (10):
    4.  Employment timeline validation: >3 concurrent overlapping jobs.
    5.  Duration integrity: jobs with negative or >600-month duration.
    6.  Job title vs responsibility consistency: AI title, 0 AI keywords in any desc.
    7.  Skill-experience consistency: advanced+proficiency with 0 months usage.
    8.  Education timeline: degree end_year before age 18.
    9.  Career progression: >3 title-level jumps in <2 years (implausible promotions).
    10. Achievement validation: 3+ inflation keywords with no metrics.
    11. Technology age: skill used before it was released.
    12. Synthetic profile: 8+ AI skills + <4 YoE + <3 jobs + high completeness.
    13. Cross-field consistency: claims NLP title + 0 NLP in skills/career.
    14. NLP-claim-without-evidence: "no NLP" in summary, but NLP skills (the contradiction).
    """
    # ----- 1. Career timeline anomaly -----
    yoe = profile.get("years_of_experience", 0) or 0
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
            if yoe > span_years + config.HONEYPOT_YOE_BUFFER_YEARS:
                features.career_timeline_anomaly = True

    # ----- 2. Expert with zero duration -----
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

    # ----- 3. Title-skills-history mismatch -----
    title_lc = (profile.get("current_title", "") or "").lower()
    has_ai_title = any(
        kw in title_lc for kw in [
            "ai engineer", "ml engineer", "data scientist", "nlp",
            "machine learning", "deep learning", "research engineer",
            "applied scientist", "ai/ml",
        ]
    )
    if has_ai_title:
        if (
            not features.has_foundational_ml
            and not features.has_ranking_skills
            and not features.has_ai_title_in_history
            and features.ai_skill_count == 0
        ):
            features.title_skills_history_mismatch = True

    # ----- 4. Employment overlap (>5 concurrent jobs at any time) -----
    if career_history and len(career_history) >= config.HONEYPOT_OVERLAP_MAX_JOBS + 1:
        intervals = []
        for c in career_history:
            s = c.get("start_date", "")
            e = c.get("end_date", "") or ""
            try:
                s_date = date.fromisoformat(s) if s else None
            except (ValueError, TypeError):
                s_date = None
            if s_date is None:
                continue
            if not e:
                e_date = date.today()
            else:
                try:
                    e_date = date.fromisoformat(e)
                except (ValueError, TypeError):
                    e_date = date.today()
            intervals.append((s_date, e_date))
        if len(intervals) >= config.HONEYPOT_OVERLAP_MAX_JOBS + 1:
            intervals.sort()
            max_concurrent = 0
            for i, (s, e) in enumerate(intervals):
                concurrent = sum(1 for s2, e2 in intervals if s2 <= s <= e2)
                max_concurrent = max(max_concurrent, concurrent)
            if max_concurrent > config.HONEYPOT_OVERLAP_MAX_JOBS:
                features.employment_overlap_anomaly = True

    # ----- 5. Duration integrity (negative or >50yr) -----
    # Stricter: only flag truly impossible (negative or >50yr), not just
    # end_date < start_date (which can be a data-entry quirk).
    for c in career_history:
        dur = c.get("duration_months", 0) or 0
        if isinstance(dur, (int, float)):
            if dur < 0 or dur > 600:  # 50 years at one company
                features.duration_integrity_violation = True
                break
        # Also check end_date < start_date (only when both are valid dates).
        s = c.get("start_date", "")
        e = c.get("end_date", "") or ""
        if s and e:
            try:
                sd = date.fromisoformat(s)
                ed = date.fromisoformat(e)
                if ed < sd:
                    features.duration_integrity_violation = True
                    break
            except (ValueError, TypeError):
                pass

    # ----- 6. Job title vs responsibility consistency -----
    # If the current title claims AI/ML but no AI/ML keywords in any job desc.
    # Stricter: only flag when title is *strongly* AI (e.g., "AI Engineer",
    # "ML Engineer") AND zero AI keywords across ALL job descriptions.
    # We exclude data-science-only roles (e.g., "Data Scientist" is borderline).
    if has_ai_title and career_history:
        # Require a strong AI/ML title (not just "Data Scientist")
        strong_ai_titles = ["ai engineer", "ml engineer", "machine learning engineer",
                            "deep learning engineer", "ai/ml engineer"]
        is_strong_ai_title = any(t in title_lc for t in strong_ai_titles)
        if is_strong_ai_title:
            ai_resp_count = 0
            for c in career_history:
                desc = (c.get("description") or "").lower()
                for kw in config.CAREER_AI_KEYWORDS:
                    if kw in desc:
                        ai_resp_count += 1
                        break
            if ai_resp_count == 0:
                features.title_responsibility_mismatch = True

    # ----- 7. Skill-experience consistency (advanced+ with 0 months) -----
    # Stricter: 5+ skills (was 3) with advanced+ and near-zero duration AND
    # zero endorsements (a real expert would have endorsements).
    adv_no_dur = 0
    for s in skills:
        if not isinstance(s, dict):
            continue
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months", 0) or 0
        endorsements = s.get("endorsements", 0) or 0
        if (
            prof in ("advanced", "expert")
            and (not isinstance(dur, (int, float)) or dur < 3)
            and (not isinstance(endorsements, (int, float)) or endorsements == 0)
        ):
            adv_no_dur += 1
    if adv_no_dur >= config.HONEYPOT_SKILL_EXP_CONTRADICTION_MIN:
        features.skill_experience_contradiction = True

    # ----- 8. Education timeline (degree before age 18) -----
    education = profile.get("education", []) or []
    if not education and career_history:
        # try candidate's education (it's at top level)
        pass
    # We need education from outside profile (passed in differently).
    # Skip here — handled in extract_features via separate call.

    # ----- 9. Career progression (3+ level jumps in 1 year, same track) -----
    # Same-track means both titles match the engineering ladder or both
    # match the management ladder. Cross-track moves (engineer → manager)
    # are not "promotions", they're role changes.
    ENG_LEVELS = {
        "intern": 0, "junior": 1, "associate": 1, "engineer": 2, "senior": 3,
        "staff": 4, "principal": 5, "fellow": 6, "distinguished": 6,
    }
    MGT_LEVELS = {
        "manager": 0, "senior manager": 1, "director": 2, "senior director": 3,
        "vp": 4, "head": 2, "chief": 4, "lead": 0, "team lead": 0,
    }
    def _track_and_level(title: str):
        t = title.lower()
        eng = max((v for k, v in ENG_LEVELS.items() if k in t), default=None)
        mgt = max((v for k, v in MGT_LEVELS.items() if k in t), default=None)
        if eng is not None and (mgt is None or eng >= mgt):
            return "eng", eng
        if mgt is not None:
            return "mgt", mgt
        return None, 0

    if career_history and len(career_history) >= 3:
        sorted_jobs = []
        for c in career_history:
            s = c.get("start_date", "")
            try:
                sd = date.fromisoformat(s) if s else None
            except (ValueError, TypeError):
                sd = None
            if sd:
                sorted_jobs.append((sd, (c.get("title") or "").lower()))
        sorted_jobs.sort()
        # Slide a 1-year window and count level jumps within the same track.
        for i in range(len(sorted_jobs)):
            base_date, base_title = sorted_jobs[i]
            base_track, base_level = _track_and_level(base_title)
            if base_track is None:
                continue
            for j in range(i + 1, len(sorted_jobs)):
                jd, jt = sorted_jobs[j]
                if (jd - base_date).days > 365:  # 1 year
                    break
                jt_track, jl = _track_and_level(jt)
                if jt_track != base_track:
                    continue  # cross-track = not a promotion
                if jl - base_level >= 3:  # 3+ levels in 1yr = implausible
                    features.career_progression_anomaly = True
                    break
            if features.career_progression_anomaly:
                break

    # ----- 10. Achievement validation (5+ inflation markers, no numbers) -----
    summary = (profile.get("summary") or "").lower()
    all_text = summary
    for c in career_history:
        all_text += " " + (c.get("description") or "").lower()
    inflation_hits = sum(1 for kw in config.HONEYPOT_ACHIEVEMENT_INFLATION_KEYWORDS if kw in all_text)
    # Check for any numeric metric (%, x improvement, scaling numbers)
    import re as _re
    has_numbers = bool(_re.search(r"\b\d+(\.\d+)?%|\b\d+x\s+(faster|improvement|more)|\b\d+\s*(users|customers|requests|qps|rps)", all_text))
    if inflation_hits >= config.HONEYPOT_ACHIEVEMENT_INFLATION_MIN and not has_numbers:
        features.achievement_inflation = True

    # ----- 11. Technology age (skill before release) -----
    # A skill is "anachronistic" if the candidate could not plausibly have
    # used it: their career started well before the tech was released.
    # Stricter: require 3+ years of pre-release, not 1.
    if career_history:
        earliest_year = None
        for c in career_history:
            s = c.get("start_date", "")
            try:
                sd = date.fromisoformat(s) if s else None
            except (ValueError, TypeError):
                sd = None
            if sd:
                earliest_year = sd.year if earliest_year is None else min(earliest_year, sd.year)
        if earliest_year:
            current_year = date.today().year
            for s in skills:
                if not isinstance(s, dict):
                    continue
                name = (s.get("name") or "").lower().strip()
                dur_months = s.get("duration_months", 0) or 0
                if isinstance(dur_months, (int, float)) and dur_months > 0:
                    skill_start_year = current_year - int(dur_months // 12)
                else:
                    skill_start_year = earliest_year
                skill_start_year = max(skill_start_year, earliest_year)
                for tech, release_year in config.TECH_RELEASE_YEARS.items():
                    if tech in name:
                        # Flag only if the skill window is clearly before release.
                        if skill_start_year < release_year - config.HONEYPOT_TECH_AGE_GRACE_YEARS:
                            features.technology_age_anomaly = True
                            break
                if features.technology_age_anomaly:
                    break

    # ----- 12. Synthetic profile (12+ AI skills, <2 YoE, 1 job, 90+ completeness) -----
    # Stricter thresholds: only the most obviously-fake profiles.
    if (
        features.ai_skill_count >= config.HONEYPOT_SYNTHETIC_PROFILE_MIN_AI_SKILLS
        and yoe < config.HONEYPOT_SYNTHETIC_PROFILE_MAX_YOE
        and len(career_history) <= config.HONEYPOT_SYNTHETIC_PROFILE_MAX_HISTORY
        and features.profile_completeness >= config.HONEYPOT_SYNTHETIC_PROFILE_MIN_COMPLETENESS
    ):
        features.synthetic_profile = True

    # ----- 13. Cross-field consistency (NLP title + no NLP anywhere) -----
    if "nlp" in title_lc or "natural language" in title_lc:
        # Check skills for any NLP keyword
        has_nlp_skill = False
        for s in skills:
            if not isinstance(s, dict):
                continue
            n = (s.get("name") or "").lower()
            if any(kw in n for kw in config.NLP_KEYWORDS):
                has_nlp_skill = True
                break
        # Check career descs
        has_nlp_career = False
        for c in career_history:
            desc = (c.get("description") or "").lower()
            if any(kw in desc for kw in config.NLP_KEYWORDS):
                has_nlp_career = True
                break
        if not has_nlp_skill and not has_nlp_career:
            features.cross_field_inconsistency = True

    # ----- 14. NLP-claim-without-evidence (the contradiction case) -----
    # User said: "candidate was saying he has no experience in NLP and want to
    # transition into it. Yet we selected him in the top 100 and gave reason that
    # he knows NLP." This is a self-contradiction.
    # Detection: summary explicitly says "no experience in NLP" / "no NLP" / "no background
    # in NLP" but skills list NLP, OR vice versa.
    summary_lc = summary
    nlp_negation_patterns = [
        "no experience in nlp", "no nlp experience", "no background in nlp",
        "no nlp background", "new to nlp", "transition into nlp",
        "transition to nlp", "no hands-on nlp", "no nlp",
    ]
    says_no_nlp = any(pat in summary_lc for pat in nlp_negation_patterns)
    has_nlp_skill = False
    for s in skills:
        if not isinstance(s, dict):
            continue
        n = (s.get("name") or "").lower()
        if any(kw in n for kw in config.NLP_KEYWORDS):
            has_nlp_skill = True
            break
    if says_no_nlp and has_nlp_skill:
        features.nlp_claim_without_evidence = True


def _detect_education_timeline_honeypot(candidate: dict, profile: dict, features: Features) -> None:
    """Detect education-timeline anomalies.

    Education lives at the candidate top-level (not inside profile). We only
    flag truly impossible timelines:
    - Degree end_year in the future.
    - Degree start_year > end_year.
    - Degree end_year before the candidate could have plausibly finished
      (e.g. end_year - birth_year < 18, only if birth_year is given).

    We do NOT flag "career started long after degree ended" because that's a
    normal pattern in this dataset (synthetic data generation quirk) and in
    real life (career break, further study, etc.).
    """
    education = candidate.get("education") or []
    if not education:
        return
    birth_year = None
    by = candidate.get("birth_year")
    if isinstance(by, (int, float)) and by > 1900:
        birth_year = int(by)
    for e in education:
        if not isinstance(e, dict):
            continue
        end_year = e.get("end_year")
        start_year = e.get("start_year")
        if not isinstance(end_year, int):
            continue
        # Future end_year.
        if end_year > date.today().year:
            features.education_timeline_anomaly = True
            return
        # start > end (impossible).
        if isinstance(start_year, int) and start_year > end_year:
            features.education_timeline_anomaly = True
            return
        # Age-based check (only if birth_year known).
        if birth_year and end_year - birth_year < config.HONEYPOT_EDU_AGE_MIN:
            features.education_timeline_anomaly = True
            return


def _compute_pre_llm_signal(career_history: list, skills: list, features: Features) -> None:
    """Boost for pre-2020 production experience in retrieval/ranking/ML.

    JD: "people who understood retrieval and ranking before it became
    fashionable". We detect this by counting career roles that started
    before 2020 and contain retrieval/ranking/ML keywords in description.
    """
    if not career_history:
        return
    pre_llm_roles = 0
    for c in career_history:
        s = c.get("start_date", "")
        try:
            sd = date.fromisoformat(s) if s else None
        except (ValueError, TypeError):
            sd = None
        if sd is None or sd.year >= config.PRE_LLM_CUTOFF_YEAR:
            continue
        desc = (c.get("description") or "").lower()
        title = (c.get("title") or "").lower()
        combined = desc + " " + title
        # Pre-LLM signals: classic IR, classical ML, embeddings, ranking.
        keywords = (
            config.CAREER_AI_KEYWORDS
            + ["information retrieval", "search engine", "search ranking",
               "relevance", "learning to rank", "bm25", "tf-idf", "tfidf",
               "word2vec", "glove", "fasttext", "lda", "topic modeling",
               "collaborative filtering", "matrix factorization", "embeddings",
               "word embeddings", "doc2vec", "elasticsearch", "solr", "lucene",
               "recommender", "recommendation system"]
        )
        if any(kw in combined for kw in keywords):
            pre_llm_roles += 1
    features.pre_llm_roles = pre_llm_roles
    # Normalize: 2+ pre-LLM roles = max signal.
    raw = min(1.0, pre_llm_roles / 2.0)
    features.pre_llm_signal = raw * config.PRE_LLM_BOOST_MAX


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

    # Education-timeline honeypot (education lives at top level, not profile)
    _detect_education_timeline_honeypot(candidate, profile, f)

    # Pre-LLM signal (pre-2020 retrieval/ranking production work)
    _compute_pre_llm_signal(career_history, skills, f)

    return f
