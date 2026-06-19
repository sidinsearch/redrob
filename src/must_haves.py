"""must_haves.py — Detect the JD's four must-haves from career_history.

Per audit spec Step 2:
  1. Production embeddings-based retrieval
  2. Production vector DB / hybrid search
  3. Eval framework design for ranking
  4. Strong Python in a real system

Detection rules:
- Evidence comes from career_history.description, NEVER the skills list.
- A skill list keyword without a career_history sentence that demonstrates
  the work does NOT satisfy the must-have.
- If a description says "platform team handled it" or similar, the work
  is NOT counted as the candidate's own.
- The detector returns a (count, evidence_dict) pair so the scoring
  layer can both count and cite specific evidence in the reasoning.

ponytail: this module is pure. Given a candidate, it returns structured
evidence. No I/O, no globals. The scoring layer reads this and assigns
the must-have baseline.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict

import config


def _gather_evidence_text(career_history: list) -> List[Tuple[str, str]]:
    """Return [(sentence, source_description), ...] for all career descriptions.

    We split each description into sentences so the reasoning layer can
    cite a specific sentence that demonstrates the must-have.
    """
    out: List[Tuple[str, str]] = []
    for job in career_history or []:
        desc = job.get("description") or ""
        if not desc:
            continue
        # Split on sentence boundaries (., ;, !, ?) but keep them.
        for sent in re.split(r"(?<=[.!?;])\s+", desc):
            sent = sent.strip()
            if sent:
                out.append((sent, desc))
    return out


def _sentence_matches(sentence: str, patterns: List[str]) -> bool:
    """True if any pattern is found in the sentence (case-insensitive)."""
    s_lower = sentence.lower()
    return any(p.lower() in s_lower for p in patterns)


def _has_disqualifying_exclusion(sentence: str, exclusions: List[str]) -> bool:
    """True if the sentence contains a phrase that disqualifies the must-have.

    Example: "production deployment was handled by the platform team" — the
    candidate did NOT do the production deployment themselves.
    """
    s_lower = sentence.lower()
    return any(e.lower() in s_lower for e in exclusions)


def _detect_must_have(
    name: str,
    sentences: List[Tuple[str, str]],
    primary: List[str],
    context: List[str],
    exclusions: List[str],
) -> Dict:
    """Detect one must-have across all career descriptions.

    Returns a dict with:
      - met: bool
      - evidence_sentences: list of specific sentences that demonstrate it
      - disqualifying_sentences: list of sentences that EXCLUDE it
        (e.g., "platform team handled it")
    """
    evidence: List[str] = []
    disqualifying: List[str] = []
    for sent, _src in sentences:
        # Exclusions are checked first — they cancel out any evidence in
        # the same sentence.
        if _has_disqualifying_exclusion(sent, exclusions):
            disqualifying.append(sent)
            continue
        if _sentence_matches(sent, primary):
            evidence.append(sent)
        elif _sentence_matches(sent, context):
            # Context hits are weak; require at least 2 context hits to count
            # as full evidence (so a single mention of "python" is not enough
            # for the strong_python must-have).
            evidence.append(sent)
    # A must-have is "met" if we have at least 1 primary hit OR at least 2
    # context hits. Single context hits are too weak (a one-word mention).
    n_primary = sum(1 for s in evidence if _sentence_matches(s, primary))
    n_context = sum(1 for s in evidence if _sentence_matches(s, context) and not _sentence_matches(s, primary))
    met = (n_primary >= 1) or (n_context >= 2)
    # If we have a disqualifying sentence but no strong evidence, must-have is NOT met
    if disqualifying and n_primary == 0:
        met = False
    return {
        "met": met,
        "evidence_sentences": evidence[:3],  # cap for reasoning
        "disqualifying_sentences": disqualifying[:2],
        "n_primary": n_primary,
        "n_context": n_context,
    }


def detect_must_haves(candidate: dict) -> Dict[str, Dict]:
    """Detect the four must-haves for a candidate.

    Returns a dict mapping must-have name → detection result.
    """
    career = candidate.get("career_history", []) or []
    sentences = _gather_evidence_text(career)
    # Pre-build a single lowercase blob for substring checks.
    # Many of the patterns overlap (e.g., "vector search" matches both
    # embeddings_retrieval and vector_db), so we check each pattern against
    # this blob once and reuse the result for all must-haves.
    blob_lower = " ".join(s.lower() for s, _ in sentences)
    out: Dict[str, Dict] = {}
    for mh_name, mh_patterns in config.MUST_HAVE_PATTERNS.items():
        out[mh_name] = _detect_must_have_fast(
            sentences,
            blob_lower,
            mh_patterns.get("primary", []),
            mh_patterns.get("context", []),
            mh_patterns.get("exclusions", []),
        )
    return out


def _detect_must_have_fast(
    sentences: List[Tuple[str, str]],
    blob_lower: str,
    primary: List[str],
    context: List[str],
    exclusions: List[str],
) -> Dict:
    """Faster detection: check patterns against a pre-built lowercase blob.

    Returns a dict with met, evidence_sentences, disqualifying_sentences,
    n_primary, n_context.
    """
    # Find which sentences have evidence. We need to track which patterns
    # matched in which sentence for the per-sentence citation.
    primary_evidence: List[str] = []
    context_evidence: List[str] = []
    disqualifying: List[str] = []
    primary_lower = [p.lower() for p in primary]
    context_lower = [p.lower() for p in context]
    exclusions_lower = [e.lower() for e in exclusions]
    n_primary = 0
    n_context = 0
    for sent, _src in sentences:
        s_lower = sent.lower()
        if any(e in s_lower for e in exclusions_lower):
            disqualifying.append(sent)
            continue
        is_primary = any(p in s_lower for p in primary_lower)
        is_context = any(p in s_lower for p in context_lower)
        if is_primary:
            primary_evidence.append(sent)
            n_primary += 1
        elif is_context:
            context_evidence.append(sent)
            n_context += 1
    met = (n_primary >= 1) or (n_context >= 2)
    if disqualifying and n_primary == 0:
        met = False
    return {
        "met": met,
        "evidence_sentences": (primary_evidence + context_evidence)[:3],
        "disqualifying_sentences": disqualifying[:2],
        "n_primary": n_primary,
        "n_context": n_context,
    }


def must_haves_met(candidate: dict) -> int:
    """Return count of must-haves met (0-4)."""
    result = detect_must_haves(candidate)
    return sum(1 for v in result.values() if v["met"])


def cite_must_have_evidence(candidate: dict, must_have_name: str) -> str:
    """Return a short evidence string for a must-have, for the reasoning field.

    Returns 'No specific evidence found in career_history.' if not met.
    """
    result = detect_must_haves(candidate)
    info = result.get(must_have_name, {})
    if info.get("met"):
        sent = (info.get("evidence_sentences") or [""])[0]
        return sent[:200]
    disqualifying = info.get("disqualifying_sentences") or []
    if disqualifying:
        return f"Disqualified: {disqualifying[0][:150]}"
    return "No specific evidence found in career_history."


# ---------------------------------------------------------------------------
# Hard disqualifiers (Step 1 of audit spec)
# ---------------------------------------------------------------------------

def is_research_only_no_production(candidate: dict) -> bool:
    """True if entire career is in pure research/academic roles.

    Detected by: every job's title contains research/academic keywords AND
    no job description contains production/deployment evidence.
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return False
    research_titles = (
        "research scientist", "research engineer", "postdoc", "post-doc",
        "research associate", "research fellow", "phd", "academic",
        "lab researcher", "research intern", "lab manager",
    )
    has_research_title = False
    has_production = False
    for job in career:
        title = (job.get("title") or "").lower()
        desc = (job.get("description") or "").lower()
        if any(rt in title for rt in research_titles):
            has_research_title = True
        if any(kw in desc for kw in (
            "production", "deployed", "shipped", "users", "serving",
            "real-time", "real time", "at scale", "in production",
        )):
            has_production = True
    return has_research_title and not has_production


def is_consulting_only_no_prior_product(candidate: dict) -> bool:
    """True if every job is at a consulting firm AND no product-company history.

    Per JD: 'People who have only worked at consulting firms in their entire
    career ... If you're currently at one of these companies but have prior
    product-company experience, that's fine.'
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return False
    has_product = False
    for job in career:
        company = (job.get("company") or "").lower().strip()
        is_consulting = any(
            c.lower() in company or company in c.lower()
            for c in config.CONSULTING_COMPANIES
        )
        if not is_consulting:
            # Found a non-consulting employer — has product company experience
            has_product = True
            break
    if has_product:
        return False
    # All employers are consulting AND we have at least 1 career entry
    return len(career) > 0


def is_cv_speech_robotics_primary(candidate: dict) -> bool:
    """True if primary expertise is CV/speech/robotics with no NLP/IR work.

    Heuristic: any career title is in CV/speech/robotics AND no career
    description or skill matches NLP/IR/retrieval keywords.
    """
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    if not career:
        return False
    cv_titles = (
        "computer vision", "cv engineer", "speech", "robotics", "autonomous",
        "image", "vision engineer", "speech scientist", "robot",
    )
    nlp_keywords = (
        "nlp", "natural language", "language model", "transformer", "llm",
        "rag", "langchain", "sentence-transformer", "retrieval", "ranking",
        "embedding", "search", "elasticsearch", "faiss", "pinecone",
    )
    has_cv_title = False
    has_nlp_evidence = False
    for job in career:
        title = (job.get("title") or "").lower()
        desc = (job.get("description") or "").lower()
        if any(ct in title for ct in cv_titles):
            has_cv_title = True
        if any(nk in desc for nk in nlp_keywords):
            has_nlp_evidence = True
    for s in skills:
        if not isinstance(s, dict):
            continue
        n = (s.get("name") or "").lower()
        if any(nk in n for nk in nlp_keywords):
            has_nlp_evidence = True
    return has_cv_title and not has_nlp_evidence


def is_langchain_only_under_12mo(candidate: dict) -> bool:
    """True if AI experience is < 12 months and primarily LangChain-only.

    Heuristic: candidate has LangChain skill + total AI-related experience
    (sum of AI skill duration_months) is < 12 months.
    """
    skills = candidate.get("skills", []) or []
    has_langchain = False
    ai_months = 0
    ai_keywords = (
        "machine learning", "deep learning", "nlp", "llm", "rag",
        "langchain", "llamaindex", "embedding", "transformer",
        "pytorch", "tensorflow", "neural",
    )
    for s in skills:
        if not isinstance(s, dict):
            continue
        n = (s.get("name") or "").lower()
        dur = s.get("duration_months", 0) or 0
        if "langchain" in n:
            has_langchain = True
        if any(ak in n for ak in ai_keywords):
            ai_months += int(dur) if isinstance(dur, (int, float)) else 0
    return has_langchain and ai_months < 12


def is_senior_no_production_code_18mo(candidate: dict) -> bool:
    """True if a senior engineer who hasn't written production code in 18+ months.

    Heuristic: current title is senior/staff/principal/lead AND most
    recent job description contains 'architecture', 'tech lead',
    'mentoring', 'no coding', or 'no production code'.
    """
    profile = candidate.get("profile", {}) or {}
    title = (profile.get("current_title") or "").lower()
    senior_kw = ("senior", "staff", "principal", "lead", "head", "vp", "director", "chief")
    if not any(sk in title for sk in senior_kw):
        return False
    career = candidate.get("career_history", []) or []
    if not career:
        return False
    # Check most recent job description
    current_job = career[0]
    desc = (current_job.get("description") or "").lower()
    no_code_phrases = (
        "no production code", "no coding", "focused on architecture",
        "architecture only", "mentoring only", "people management only",
        "no implementation", "no longer writing code",
    )
    return any(p in desc for p in no_code_phrases)


def apply_hard_disqualifiers(candidate: dict) -> Tuple[bool, str]:
    """Return (disqualified, reason) for hard disqualifiers (Step 1).

    If disqualified, the candidate's fit_score should be 0.
    """
    if is_research_only_no_production(candidate):
        return True, "research-only-no-production"
    if is_consulting_only_no_prior_product(candidate):
        return True, "consulting-only-no-prior-product"
    if is_cv_speech_robotics_primary(candidate):
        return True, "cv-speech-robotics-primary"
    if is_langchain_only_under_12mo(candidate):
        return True, "langchain-only-under-12mo"
    if is_senior_no_production_code_18mo(candidate):
        return True, "senior-no-production-code-18mo"
    return False, ""


# ---------------------------------------------------------------------------
# Negative adjustments (subtract from baseline)
# ---------------------------------------------------------------------------

def detect_negative_patterns(candidate: dict) -> List[str]:
    """Return a list of negative pattern names that fired for this candidate.

    Each name in the list corresponds to a config.NEGATIVE_PATTERNS key.
    """
    sentences = _gather_evidence_text(candidate.get("career_history", []) or [])
    all_text = " ".join(s for s, _ in sentences).lower()
    summary = (candidate.get("profile", {}).get("summary") or "").lower()
    combined = all_text + " " + summary
    out: List[str] = []
    for name, patterns in config.NEGATIVE_PATTERNS.items():
        if any(p.lower() in combined for p in patterns):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Nice-to-haves (+1 to +3 each, max +10 total)
# ---------------------------------------------------------------------------

def detect_nice_to_haves(candidate: dict) -> Dict[str, int]:
    """Return a dict mapping nice-to-have name → number of evidence hits."""
    sentences = _gather_evidence_text(candidate.get("career_history", []) or [])
    all_text = " ".join(s for s, _ in sentences).lower()
    out: Dict[str, int] = {}
    for name, patterns in config.NICE_TO_HAVE_PATTERNS.items():
        hits = sum(1 for p in patterns if p.lower() in all_text)
        if hits > 0:
            out[name] = min(3, hits)  # cap at 3 per category
    return out
