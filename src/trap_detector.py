"""trap_detector.py — Classify candidates into 4 trap types and honeypots.

Each trap is independently detected. Multiple traps apply multiplicatively
(floored at TRAP_FLOOR). Honeypots are forced to score 0.0 — they cannot
appear in the top 100.

The trap detection is the single most important stage for Stage 3 survival.
A 10% honeypot rate triggers disqualification. We aim for 0%.

ponytail: this module is pure — no I/O. Given Features in, returns trap info.
Trap detection must run BEFORE scoring, so we can apply the multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from features import Features


# ----------------------------------------------------------------------------
# Trap type detection
# ----------------------------------------------------------------------------

def detect_keyword_stuffer(f: Features) -> bool:
    """Type 1: Non-technical title + 4+ AI skills + zero foundational ML +
    skills are only LLM buzzwords.

    This is the explicit trap from the JD: marketing/HR/finance people who
    list RAG, LangChain, Embeddings as skills but have never built anything.
    """
    if f.title_score >= 0.40:
        return False  # Has a real title
    if f.ai_skill_count < config.KEYWORD_STUFFER_MIN_AI_SKILLS:
        return False
    if f.has_foundational_ml:
        return False  # Real ML — not a stuffer
    if f.has_ranking_skills:
        return False
    if f.has_ai_title_in_history:
        return False
    # Check that buzzwords dominate the AI skills
    if not f.has_llm_buzzwords_only:
        return False
    return True


def detect_template_summary(f: Features) -> bool:
    """Type 2: Summary contains the canned 'curious about AI tools' phrase.

    The dataset has ~63K candidates with this exact phrase. It's a strong
    signal of a generic AI-curious non-specialist.
    """
    return f.template_summary_match


def detect_consulting_only(f: Features) -> bool:
    """
    Type 3: Candidate has worked ONLY at consulting/services companies.

    JD:
    - Reject candidates whose entire career consists only of consulting firms
      (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.).
    - If the candidate has worked at even one product company anywhere in
      their career, do NOT flag them.
    """

    # No work history -> cannot determine
    if not f.career_history:
        return False

    # If feature extraction already detected any product company,
    # the candidate should NOT be flagged.
    if f.career_product_ratio > 0.0:
        return False

    # Every employer must be a consulting company.
    for job in f.career_history:
        company = (job.get("company") or "").lower().strip()

        is_consulting = any(
            consulting.lower() in company or company in consulting.lower()
            for consulting in config.CONSULTING_COMPANIES
        )

        # Found a company that is not consulting.
        if not is_consulting:
            return False

    # All employers are consulting firms.
    return True


def detect_title_chaser(f: Features) -> bool:
    """Type 4: 4+ jobs in last 8 years with avg tenure <18 months.

    Per the JD: 'Title-chasers. If your career trajectory shows you optimizing
    for Senior → Staff → Principal titles by switching companies every 1.5
    years, we're not a fit.'
    """
    return f.is_title_chaser


# ----------------------------------------------------------------------------
# Honeypot detection
# ----------------------------------------------------------------------------

# Spec patterns (must-have, derived from submission_spec.docx):
#   - career_timeline_anomaly: "8 years of experience at a company founded 3 years ago"
#   - expert_with_zero_duration: "expert proficiency in 10 skills with 0 years used"
#   - title_skills_history_mismatch: "AI title with no AI skills or AI history"
#
# Additional patterns (3 most relevant from the 10 requested):
#   - title_responsibility_mismatch: AI/ML title but 0 AI/ML keywords in any
#     job description across the entire career.
#   - technology_age_anomaly: claims expertise in a technology that did not
#     exist when the candidate's career started.
#   - cross_field_inconsistency: title claims NLP/IR/CV but 0 of those in
#     skills or career (e.g., "NLP Engineer" with no NLP anywhere).
#
# Other detectors (employment_overlap, duration_integrity, skill_experience,
# education_timeline, career_progression, achievement_inflation,
# synthetic_profile, nlp_claim_without_evidence) are present as safety nets
# with strict thresholds. They are dormant on this dataset but ready to fire
# on truly impossible profiles.


def is_honeypot(f: Features) -> bool:
    """Return True if candidate is a forced-zero honeypot.

    We check 14 patterns. Any single positive flag forces the candidate to
    the bottom of the ranking. Reasons are reported in honeypot_reasons.

    ponytail: 14 flags is more than we strictly need. The 3 spec patterns
    are the contract; the 3 most relevant from the user's list
    (title_responsibility, technology_age, cross_field) are the practical
    additions. The other 8 are dormant safety nets.
    """
    return any([
        # Spec patterns (3)
        f.career_timeline_anomaly,
        f.expert_with_zero_duration,
        f.title_skills_history_mismatch,
        # 3 most relevant from the 10 user-requested detectors
        f.title_responsibility_mismatch,
        f.technology_age_anomaly,
        f.cross_field_inconsistency,
        # Dormant safety nets (8) — strict thresholds, ready to fire on
        # truly impossible profiles if any exist in the data
        f.employment_overlap_anomaly,
        f.duration_integrity_violation,
        f.skill_experience_contradiction,
        f.education_timeline_anomaly,
        f.career_progression_anomaly,
        f.achievement_inflation,
        f.synthetic_profile,
        f.nlp_claim_without_evidence,
    ])


# ----------------------------------------------------------------------------
# Combined trap analysis
# ----------------------------------------------------------------------------

@dataclass
class TrapInfo:
    """Result of trap detection for a single candidate."""
    is_honeypot: bool
    is_keyword_stuffer: bool
    is_template_summary: bool
    is_consulting_only: bool
    is_title_chaser: bool
    trap_multiplier: float
    honeypot_reasons: list  # human-readable


def analyze(f: Features) -> TrapInfo:
    """Run all trap detectors. Returns a TrapInfo with multiplier."""
    is_honeypot_flag = is_honeypot(f)
    is_ks = detect_keyword_stuffer(f)
    is_ts = detect_template_summary(f)
    is_co = detect_consulting_only(f)
    is_tc = detect_title_chaser(f)

    # If keyword stuffer, skip template summary (already covered)
    if is_ks:
        is_ts = False

    # Multiplier
    if is_honeypot_flag:
        multiplier = config.HONEYPOT_TAX
    else:
        multiplier = 1.0
        if is_ks:
            multiplier *= config.TRAP_MULTIPLIERS["keyword_stuffer"]
        if is_ts:
            multiplier *= config.TRAP_MULTIPLIERS["template_summary"]
        if is_co:
            multiplier *= config.TRAP_MULTIPLIERS["consulting_only"]
        if is_tc:
            multiplier *= config.TRAP_MULTIPLIERS["title_chaser"]
        # Floor (except for honeypots which are forced to bottom)
        multiplier = max(multiplier, config.TRAP_FLOOR)

    # Reasons — 13 honeypot patterns
    reasons = []
    if f.career_timeline_anomaly:
        reasons.append("career timeline impossible (YoE > career span)")
    if f.expert_with_zero_duration:
        reasons.append(f"expert in {config.EXPERT_SKILL_FAKE_THRESHOLD}+ skills with 0 months use")
    if f.title_skills_history_mismatch:
        reasons.append("AI title with no AI skills or AI history")
    if f.employment_overlap_anomaly:
        reasons.append(">3 concurrent overlapping jobs")
    if f.duration_integrity_violation:
        reasons.append("invalid job duration (negative or >50yr)")
    if f.title_responsibility_mismatch:
        reasons.append("AI/ML title but 0 AI/ML keywords in any job description")
    if f.skill_experience_contradiction:
        reasons.append("advanced+ proficiency in 3+ skills with near-zero duration")
    if f.education_timeline_anomaly:
        reasons.append("education timeline impossible (degree before age 18 or after career start)")
    if f.career_progression_anomaly:
        reasons.append("implausible career progression (3+ title-level jumps in 2yr)")
    if f.achievement_inflation:
        reasons.append(f"{config.HONEYPOT_ACHIEVEMENT_INFLATION_MIN}+ inflation keywords with no metrics")
    if f.technology_age_anomaly:
        reasons.append("skill expertise predates the technology's release")
    if f.synthetic_profile:
        reasons.append("synthetic profile (many AI skills, low YoE, few jobs)")
    if f.cross_field_inconsistency:
        reasons.append("field mismatch (e.g. NLP title with 0 NLP in skills/career)")
    if f.nlp_claim_without_evidence:
        reasons.append("says no NLP experience but lists NLP skills (self-contradiction)")
    if is_ks:
        reasons.append("keyword stuffer (no foundational ML)")
    if is_ts:
        reasons.append("canned AI-curious summary")
    if is_co:
        reasons.append("consulting-only career")
    if is_tc:
        reasons.append(f"title-chaser ({f.num_jobs_8yr} jobs in 8yr, avg {f.avg_tenure_months_8yr:.0f}mo)")

    return TrapInfo(
        is_honeypot=is_honeypot_flag,
        is_keyword_stuffer=is_ks,
        is_template_summary=is_ts,
        is_consulting_only=is_co,
        is_title_chaser=is_tc,
        trap_multiplier=multiplier,
        honeypot_reasons=reasons,
    )
