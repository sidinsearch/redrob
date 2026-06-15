# Methodology — How to Defend This Ranker

This document explains the design rationale behind every weight, threshold, and
rule in the ranker. It's written to be read at Stage 5 (defend-your-work
interview) by a Redrob engineer who has 30 minutes to evaluate whether you
understand your own system.

If you can't defend a number in `src/config.py`, this document is incomplete.

---

## 1. Why stdlib-only?

The constraint is 5 minutes on CPU. Neural embeddings (sentence-transformers,
BGE, E5) cost ~50ms per candidate in batched CPU mode. For 100K candidates,
that's 5,000 seconds — already over the budget before you do anything else.

A pre-compute step moves that offline, but then:
1. Pre-compute is excluded from the 5-min window only if we can ship the
   embeddings as artifacts. The submission spec is explicit: pre-computation
   is allowed, but the *ranking step* must run within 5 minutes.
2. Embedding model versions drift. The challenge reproduction will pull
   the same model version we used. If that version becomes unavailable,
   we lose reproducibility.
3. The dataset has explicit traps (keyword stuffers, template summaries)
   that are nearly orthogonal to embedding similarity. An embedding
   model ranks "Marketing Manager who lists RAG/LangChain/Embeddings as
   skills" highly because their skill list is textually similar to the JD.

Pure rule-based scoring avoids all three. The "BM25 + rule-based, working
but not great" quote in the JD is the rubric: we beat BM25+rules on the
specific trap patterns BM25 misses, and we don't take on embedding risks.

## 2. Why these weights?

| Component | Weight | Justification |
|-----------|--------|---------------|
| title_relevance | 0.25 | The strongest single signal. The JD title is "Senior AI Engineer". A real candidate usually has a related title. |
| ai_skills_depth | 0.15 | Catches the "Tier 5" plain-language candidates who use the right tools but don't have the title. |
| behavioral_score | 0.15 | Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available." |
| experience_fit | 0.12 | The JD says "5-9 years"; we center the score at 6-8 with graceful degradation outside the band. |
| product_exp | 0.10 | The JD explicitly disqualifies consulting-only careers. |
| career_relevance | 0.10 | Catches the JD's "Tier 5" who built ranking systems without using the buzzword. |
| location_fit | 0.05 | India preferred. Small weight; not a deal-breaker for a strong candidate. |
| availability_score | 0.05 | ≤30d notice strongly preferred. |
| education_score | 0.03 | Tier-1 institution is a small positive; we don't penalize missing top-tier education because the field is full of self-taught talent. |

Total: 1.00. Verified in `config.WEIGHTS` (an assertion fails if the sum drifts).

## 3. Why a 4-level trap detection?

The dataset has four explicitly trap patterns:

1. **Keyword stuffer** — non-technical title (Marketing, HR, Sales) + 4+
   AI skills + zero foundational ML (PyTorch, TensorFlow, etc.) + skills
   are only LLM buzzwords. This is the dominant trap.

2. **Template summary** — exact phrase "Lately I've been curious about
   how AI tools could augment my work" appears in ~63K candidates' summaries.
   It's a marker of "AI-curious non-specialist" profiles.

3. **Consulting-only** — JD explicitly lists TCS/Infosys/Wipro/Accenture
   as bad-fit companies for this role. We hard-code the list and apply
   a 25% penalty.

4. **Title-chaser** — JD: "If your career trajectory shows you optimizing
   for Senior → Staff → Principal titles by switching companies every
   1.5 years, we're not a fit." 4+ jobs in 8 years with avg tenure
   <18 months triggers the flag.

Each is a 5-10 line detector in `src/trap_detector.py`. They apply
multiplicatively with a floor at 0.30 (i.e., even with all 4 traps, a
candidate retains 30% of their raw score). This is the right shape
because: a) the detectors aren't perfect (false positives cost us), and
b) the dominant signal is still the underlying quality.

## 4. Why 3 honeypot patterns?

Honeypots are the disqualification trigger. We force them to 0% of the
top 100 by detecting 3 specific patterns:

1. **Career timeline anomaly** — `YoE > (career_span + 5 years)`. Catches
   "8 years at a company founded 3 years ago" profiles.

2. **Expert with zero duration** — "expert" in 5+ skills but 0 months of
   use. Catches profiles with empty skill endorsements.

3. **Title-skills-history mismatch** — "Senior AI Engineer" title with
   0 AI skills AND no AI career history. Catches title-only profiles
   that might pass title scoring.

We force these to `score = -1e9` (effectively bottom) regardless of
their other signals. This guarantees 0% honeypot rate in the top-100.

## 5. Why these specific skills as "foundational"?

The "foundational ML" set is the discriminator between a real ML engineer
and a keyword stuffer. The set includes:

- Frameworks: PyTorch, TensorFlow, Keras, scikit-learn
- Concepts: machine learning, deep learning, neural networks, NLP
- Tasks: NDCG, MRR, MAP (eval), learning-to-rank
- Architectures: transformers, BERT-style
- Production: MLflow, Kubeflow, Docker, Kubernetes
- Search/retrieval: FAISS, Pinecone, Elasticsearch, BM25

A candidate with 4+ "AI skills" but no foundational ML has likely
copy-pasted a buzzword list. The trap detector flags them with 0.40×
multiplier.

A candidate with foundational ML + LLM buzzwords is fine — they have
both eras. Only the buzzword-only profile is the trap.

## 6. Why vary the reasoning template by rank?

Stage 4 manual review checks "rank consistency" — does a rank-5 candidate
have critical reasoning, or a rank-95 candidate have glowing reasoning?

We use 4 templates:
- **Top 10**: lead with the title and YoE, cite specific skills, connect
  to JD. Critical tone only when there's a real concern (e.g., 90d notice).
- **Top 30**: more breadth, cite career history, acknowledge the strongest
  non-behavioral signal.
- **Mid (31-60)**: tighter, fewer details, honest about being "in the band"
  rather than "the best fit".
- **Bottom (61-100)**: explicit about being borderline. Cites the limiting
  factor (consulting-only, title-chaser, lacking ML).

Each template cites only real profile fields. We never fabricate skills,
employers, or experience — the reasoning generator is fed a `Features`
dataclass and only uses fields that were explicitly extracted.

## 7. What's NOT in the ranker (and why)

- **No embeddings** — discussed in §1.
- **No LLM calls** — violates the no-network constraint.
- **No clustering / diversity reranking** — the spec scores on NDCG/MAP
  which rewards pure relevance. Diversity hurts MAP because it forces
  the model to include lower-quality candidates. We do not use MMR.
- **No personal-name matching** — the dataset has anonymized names, so
  there's no signal to extract.
- **No model retraining** — pure rules are easier to defend at Stage 5
  and don't have version drift.

## 8. Compute budget breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Stream 100K JSONL | ~7s | Pure stdlib `json.loads` |
| Feature extraction | ~25s | ~10 string contains per candidate × 40 features |
| Trap detection | ~3s | Substring matches + counters |
| Sort top-100 | ~0.5s | `list.sort` on 100K items |
| Reasoning generation | ~0.2s | Template fill × 100 |
| CSV write + validate | <0.1s | `csv.writer` |
| **Total** | **~36s** | **5x under the 5-min budget** |

Memory peaks at ~1.5 GB (we hold 100K `Features` records in lists).
Well under the 16 GB limit.

## 9. Stage-5 talking points

If asked "why not just use an LLM?":
> "Five-minute CPU budget, no network during ranking, and the dataset has
> explicit traps that an LLM would actually miss because their skill list
> looks similar to the JD. Pure rule-based scoring lets us catch those
> traps with high precision."

If asked "what's the weakest part of your system?":
> "The 0.05 score for non-technical titles. A Tier 5 candidate (per the JD)
> with a 'Business Analyst' title who built a recommendation system would
> get penalized by my title scoring. I mitigate this via the career_relevance
> and ai_skills_depth components, but a more sophisticated solution would
> learn the title-from-career mapping from labeled data."

If asked "how would you improve it with more time?":
> "Add a small learned re-ranker (e.g., XGBoost) on the 40+ features
> extracted here. Train on a small labeled set of past hires, score
> all 100K candidates with it, then use those scores as a tie-breaker
> on my rule-based top-100. The rule-based top-100 still gates on
> trap detection, so we never regress on the 0% honeypot guarantee."

If asked "why these specific consulting companies?":
> "Direct from the JD: 'People who have only worked at consulting firms
> (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their
> entire career.' I included all explicitly named firms plus a few
> common variants (Mindtree, Mphasis, LTIMindtree) that share the
> services-business model."

## 10. What we test

| Test | What it covers |
|------|----------------|
| `test_sample.py` | Full pipeline on 50-candidate sample; verifies features + traps + scoring + output |
| `test_traps.py` | Each trap type fires on a known-bad candidate; clean candidates don't trigger |
| `test_honeypots.py` | Each honeypot pattern fires on a known-bad profile |
| `test_output.py` | CSV writer produces valid format; validator accepts it |

Tests are run with `python -m pytest tests/` from the project root.
