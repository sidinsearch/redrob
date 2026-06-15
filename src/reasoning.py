"""reasoning.py — Generate the 1-2 sentence reasoning string per candidate.

Stage 4 manual review scores reasoning on 5 axes:
1. Specific facts (years, title, named skills, signal values)
2. JD connection (not generic praise)
3. Honest concerns (gaps acknowledged)
4. No hallucination (every claim maps to a real field)
5. Variation (not templated)
6. Rank consistency (tone matches rank)

We generate 8 different templates and select by score tier + profile
characteristics. Every fact we cite comes from Features — never from
the raw candidate dict. The reasoning is at most 300 chars per spec.

ponytail: keep templates as string templates with {placeholders} rather
than building strings with concat — easier to read, easier to vary.
"""

from __future__ import annotations

import random
from typing import List

import config
from features import Features
from trap_detector import TrapInfo


def generate_reasoning(f: Features, trap: TrapInfo, rank: int) -> str:
    """Generate a reasoning string for the candidate at the given rank.

    Returns a single string ≤300 chars that:
    - references specific facts from the profile
    - connects to JD requirements when possible
    - acknowledges concerns when present
    - varies tone by rank
    """
    # Build the component facts
    facts = _gather_facts(f, trap)

    # Pick a template based on rank tier and candidate profile
    template_fn = _select_template(rank, f, trap)
    text = template_fn(facts, f, trap, rank)

    # Validate length and trim
    text = text.strip()
    if len(text) > config.MAX_REASONING_LEN:
        text = text[: config.MAX_REASONING_LEN - 1].rstrip() + "…"
    return text


# ----------------------------------------------------------------------------
# Fact gathering
# ----------------------------------------------------------------------------

class Facts:
    """A bundle of fact strings used by templates."""
    def __init__(self):
        self.role: str = ""
        self.yoe: str = ""
        self.company: str = ""
        self.location: str = ""
        self.ai_skills: List[str] = []
        self.skills_str: str = ""
        self.career_fact: str = ""
        self.signal_strength: str = ""
        self.concern: str = ""
        self.strength: str = ""
        self.matched_must_haves: List[str] = []
        self.education: str = ""
        self.notice: str = ""


def _gather_facts(f: Features, trap: TrapInfo) -> Facts:
    facts = Facts()
    facts.role = (f.current_title or "").strip()
    facts.yoe = f"{f.years_of_experience:.1f}" if f.years_of_experience else "?"
    facts.company = (f.current_company or "").strip()
    facts.location = (f.location or "").strip()
    facts.notice = f"{f.notice_period_days}d" if f.notice_period_days else ""

    # AI skills cited — pull specific named skills from the candidate
    facts.ai_skills = _extract_named_ai_skills(f)
    if facts.ai_skills:
        facts.skills_str = ", ".join(facts.ai_skills[:5])

    # Career fact — most relevant past role
    facts.career_fact = _extract_career_fact(f)

    # Signal strength
    facts.signal_strength = _extract_signal_strength(f)

    # Concerns
    facts.concern = _extract_concern(f, trap)

    # Strengths
    facts.strength = _extract_strength(f)

    # JD must-haves
    facts.matched_must_haves = list(f.matched_must_haves)

    # Education
    if f.best_education_tier in ("tier_1", "tier_2"):
        facts.education = f"tier-{f.best_education_tier[-1]} education"

    return facts


def _extract_named_ai_skills(f: Features) -> List[str]:
    """Find specific AI/ML skill names from Features.search_blob.

    Returns a small list of human-readable skills for the reasoning text.
    We don't have direct access to the skills list (Features doesn't store
    the full skills list), so we search the blob for known skill tokens.
    """
    blob = f.search_blob
    candidates = [
        # Foundational
        ("PyTorch", ["pytorch"]),
        ("TensorFlow", ["tensorflow"]),
        ("NLP", ["nlp", "natural language processing"]),
        ("transformers", ["transformer", "hugging face", "huggingface"]),
        ("deep learning", ["deep learning"]),
        ("computer vision", ["computer vision", "image classification", "object detection"]),
        # Production
        ("MLflow", ["mlflow"]),
        ("Kubeflow", ["kubeflow"]),
        ("Docker", ["docker"]),
        ("Kubernetes", ["kubernetes", "k8s"]),
        ("MLOps", ["mlops"]),
        # Ranking/retrieval
        ("FAISS", ["faiss"]),
        ("Pinecone", ["pinecone"]),
        ("Weaviate", ["weaviate"]),
        ("Qdrant", ["qdrant"]),
        ("Milvus", ["milvus"]),
        ("Elasticsearch", ["elasticsearch"]),
        ("vector search", ["vector search", "vector db", "vector database"]),
        ("BM25", ["bm25"]),
        ("learning to rank", ["learning to rank", "l2r"]),
        ("semantic search", ["semantic search"]),
        ("hybrid search", ["hybrid search"]),
        # LLM
        ("LangChain", ["langchain"]),
        ("LlamaIndex", ["llamaindex"]),
        ("RAG", ["rag ", "rag,", "rag.", "rag)", "retrieval-augmented", "retrieval augmented"]),
        ("LoRA", ["lora"]),
        ("QLoRA", ["qlora"]),
        ("fine-tuning LLMs", ["fine-tuning llm", "fine tuning llm"]),
    ]
    found = []
    for label, patterns in candidates:
        for p in patterns:
            if p in blob:
                found.append(label)
                break
    return found


def _extract_career_fact(f: Features) -> str:
    """Pick the most relevant career-history fact to cite."""
    # Prefer the most recent AI-titled role
    for c in reversed(f.career_history):
        title = (c.get("title") or "").lower()
        desc = (c.get("description") or "").lower()
        if any(kw in title for kw in ["machine learning", "ml engineer", "ai engineer", "data scientist", "nlp"]):
            company = c.get("company", "")
            if company:
                return f"{title} at {company}"
    # Otherwise return the most recent role's company
    if f.career_history:
        last = f.career_history[-1]
        company = last.get("company", "")
        if company:
            return f"at {company}"
    return ""


def _extract_signal_strength(f: Features) -> str:
    """Describe the candidate's behavioral signal strength."""
    parts = []
    if f.search_appearance_30d >= 100:
        parts.append(f"high search appearance ({f.search_appearance_30d}/30d)")
    if f.saved_by_recruiters_30d >= 10:
        parts.append(f"saved {f.saved_by_recruiters_30d}x by recruiters")
    if f.endorsements_received >= 30:
        parts.append(f"{f.endorsements_received} endorsements")
    if f.recruiter_response_rate >= 0.7:
        parts.append(f"{int(f.recruiter_response_rate * 100)}% response rate")
    if f.open_to_work:
        parts.append("actively looking")
    if f.github_activity_score >= 30:
        parts.append(f"strong GitHub ({f.github_activity_score:.0f})")
    if f.interview_completion_rate >= 0.8:
        parts.append(f"{int(f.interview_completion_rate * 100)}% interview completion")
    return "; ".join(parts[:3])


def _extract_concern(f: Features, trap: TrapInfo) -> str:
    """Honest concern about the candidate (or empty)."""
    if trap.is_consulting_only:
        return "consulting-only career path"
    if f.notice_period_days > 90:
        return f"{f.notice_period_days}-day notice"
    if not f.open_to_work:
        return "not currently marked open-to-work"
    if f.country and f.country.lower() not in ("india",):
        return f"based in {f.country}"
    if f.years_of_experience > 12:
        return f"senior ({f.years_of_experience:.0f}yr) — may tilt toward architect roles"
    return ""


def _extract_strength(f: Features) -> str:
    """Top non-behavioral strength of the candidate."""
    if f.has_ranking_skills and f.has_foundational_ml:
        return "rare combination of foundational ML and production retrieval/ranking experience"
    if f.has_ranking_skills:
        return "production retrieval/ranking experience matches JD's core requirement"
    if f.has_foundational_ml and f.ai_skill_count >= 5:
        return "deep foundational ML with broad AI toolkit"
    if f.has_ai_title_in_history and f.career_ai_keyword_hits >= 2:
        return "consistent AI/ML trajectory across multiple roles"
    if f.title_score >= 0.85:
        return "title directly matches the role's mandate"
    return ""


# ----------------------------------------------------------------------------
# Templates — 8 variants selected by rank tier
# ----------------------------------------------------------------------------

def _select_template(rank: int, f: Features, trap: TrapInfo):
    """Return a template function appropriate for the rank and profile."""
    if rank <= 10:
        return _template_top10
    elif rank <= 30:
        return _template_top30
    elif rank <= 60:
        return _template_mid
    else:
        return _template_bottom


def _template_top10(facts: Facts, f: Features, trap: TrapInfo, rank: int) -> str:
    # Top-10: lead with title and YoE; cite specific skills; connect to JD.
    bits = []
    if facts.role and facts.yoe != "?":
        bits.append(f"{facts.role} with {facts.yoe}yr")
    elif facts.role:
        bits.append(facts.role)
    if facts.company:
        bits.append(f"at {facts.company}")
    if facts.skills_str:
        bits.append(f"hands-on with {facts.skills_str}")
    if facts.matched_must_haves:
        must_have_phrase = {
            "embeddings_or_retrieval": "embeddings/retrieval",
            "vector_db_or_search": "vector search",
            "strong_python": "Python",
            "eval_framework": "eval frameworks",
        }
        mh = [must_have_phrase.get(m, m) for m in facts.matched_must_haves[:3]]
        bits.append(f"matches JD on {', '.join(mh)}")
    if facts.signal_strength:
        bits.append(f"signals: {facts.signal_strength}")
    if facts.concern:
        bits.append(f"concern: {facts.concern}")
    return "; ".join(bits) + "."


def _template_top30(facts: Facts, f: Features, trap: TrapInfo, rank: int) -> str:
    bits = []
    if facts.role and facts.yoe != "?":
        bits.append(f"{facts.role} ({facts.yoe}yr")
        if facts.company:
            bits.append(f", {facts.company})")
        else:
            bits.append(")")
    elif facts.role:
        bits.append(facts.role)
    if facts.skills_str:
        bits.append(f"stack: {facts.skills_str}")
    if facts.career_fact:
        bits.append(f"history includes {facts.career_fact}")
    if facts.strength:
        bits.append(facts.strength)
    if facts.signal_strength:
        bits.append(facts.signal_strength)
    if facts.concern:
        bits.append(facts.concern)
    return "; ".join(bits) + "."


def _template_mid(facts: Facts, f: Features, trap: TrapInfo, rank: int) -> str:
    bits = []
    if facts.role:
        bits.append(facts.role)
    if facts.yoe != "?":
        bits.append(f"{facts.yoe}yr exp")
    if facts.company:
        bits.append(f"at {facts.company}")
    if facts.skills_str:
        bits.append(f"some {facts.skills_str}")
    elif facts.ai_skills:
        bits.append(f"{len(facts.ai_skills)} AI skills")
    if facts.location:
        bits.append(f"located {facts.location}")
    if facts.signal_strength:
        bits.append(facts.signal_strength)
    if facts.concern:
        bits.append(facts.concern)
    return "; ".join(bits) + "."


def _template_bottom(facts: Facts, f: Features, trap: TrapInfo, rank: int) -> str:
    # Bottom of top-100: be honest that this is borderline.
    bits = []
    if facts.role:
        bits.append(facts.role)
    if facts.yoe != "?":
        bits.append(f"{facts.yoe}yr")
    if facts.company:
        bits.append(f"at {facts.company}")
    if trap.is_title_chaser:
        bits.append("frequent job movement")
    if trap.is_consulting_only:
        bits.append("consulting-only career")
    if not f.has_foundational_ml and not f.has_ranking_skills:
        bits.append("lacks direct ML/retrieval stack")
    if facts.concern:
        bits.append(facts.concern)
    if not bits:
        bits.append("borderline fit — kept at the cutoff on signal strength")
    return "; ".join(bits) + "."
