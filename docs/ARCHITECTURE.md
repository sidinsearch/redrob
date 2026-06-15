# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

## Architecture & Execution Plan

---

## 1. PROBLEM SUMMARY

**Goal**: Rank 100K candidates against a Senior AI Engineer — Founding Team JD and output the top 100 in CSV format.

**Key JD requirements** (reading between the lines):
- 6–8 yrs total, 4–5 in applied ML at **product companies** (not pure services/consulting)
- Shipped ranking/search/recommendation systems to real users
- Production experience: embeddings, vector DBs, eval frameworks (NDCG/MRR/MAP)
- **NOT**: pure researchers, LangChain-only devs, consulting-only career (TCS/Infosys/Wipro/Accenture), CV/speech without NLP/IR
- Location: India (Pune/Noida preferred), notice period ≤30 days preferred

**Constraints**:
| Constraint | Limit |
|-----------|-------|
| Runtime | ≤5 min wall-clock (CPU only) |
| Memory | ≤16 GB RAM |
| Network | Off — no external API calls |
| GPU | None |
| Output | CSV with 100 rows |

**Scoring**:
| Metric | Weight | What it measures |
|--------|--------|-----------------|
| NDCG@10 | 50% | Quality of top-10 |
| NDCG@50 | 30% | Quality of top-50 |
| MAP | 15% | Precision across all levels |
| P@10 | 5% | Fraction of top-10 relevant |

**Hidden traps in dataset**:
1. **~8,834 keyword stuffers** — non-technical titles (Marketing, HR, Accountant) with 4+ AI buzzword skills (RAG, LangChain, Embeddings, Fine-tuning LLMs, Recommendation Systems) but zero foundational ML
2. **Template summary trap** — ~63,304 candidates have the exact phrase "Lately I've been curious about how AI tools could augment my work — I've experimented with ChatGPT..."
3. **~80 honeypots** — impossible profiles (8yr exp at company founded 3yr ago, expert in 10 skills with 0yr used)
4. **Consulting-only career** — TCS/Infosys/Wipro-only careers flagged as bad fit per JD

---

## 2. ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────────────────┐
│                     rank.py (Main Entry Point)                    │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │  Parser       │  │  Feature       │  │  Scoring Engine    │   │
│  │  (JSONL →     │→ │  Extractor     │→ │  (Rule-based +     │   │
│  │   dict list)  │  │                │  │   Weighted Sum)    │   │
│  └──────────────┘  └────────────────┘  └─────────┬──────────┘   │
│                                                  │              │
│  ┌──────────────┐  ┌────────────────┐            │              │
│  │  Trap        │  │  Honeypot      │            │              │
│  │  Detector    │  │  Detector      │────────────┘              │
│  └──────────────┘  └────────────────┘                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Reasoning Generator → Output CSV                            ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

**Design philosophy**: NOT a deep learning model — a **rule-based weighted scoring system** with explicit trap detection. This is deliberate:
- JD explicitly says "We've tried BM25 + rule-based, working but not great" — meaning rule-based is the **baseline they know**, and improvements on it are visible
- 5-min CPU constraint means no embedding computation per-candidate (that's 100K candidate * embedding cost)
- Rule-based is interpretable, debuggable, and passes Stage 5 interview
- Focus is on **identifying the right signals and weighting them correctly**, not on model complexity

---

## 3. DATA PIPELINE

### 3.1 Parse Phase (~6 seconds)
- Read `candidates.jsonl` line by line
- Parse each line as JSON → Python dict
- Store in list of 100K dicts
- Track candidate_ids for validation

### 3.2 Feature Extraction Phase (~30 seconds)
For each candidate, extract:

**A. Profile Features:**
- `current_title` → title_category (ML/Technical/Non-technical)
- `years_of_experience` → raw value + bucketed score
- `country` → India score boost
- `headline` + `summary` → check for template trap phrase
- `current_company` → consulting flag + product company flag
- `current_industry` → relevance to AI/tech

**B. Career History Features:**
- All companies → check for product company experience (any non-consulting)
- Has AI/ML role titles in history
- Has search/ranking/recommendation keywords in descriptions
- Total career span, job-hopping frequency (title-chaser detection)
- Current company → consulting or product

**C. Skills Features:**
- Count of AI/ML skills from curated skill ontology
- Flag if has foundational ML skills (PyTorch, TensorFlow, Deep Learning, NLP, Machine Learning) — **key anti-trap signal**
- Flag if has only LLM-app buzzwords (RAG, LangChain, Embeddings, Fine-tuning LLMs) without foundational skills
- Max proficiency level ("expert" vs "intermediate")
- Total skill count inflation check (>17 skills = suspicious)
- Skill assessment completed flag

**D. Education Features:**
- Institution tier (tier_1, tier_2 = strong positive)
- Field of study matches CS/AI/ML
- Degree level relevance

**E. Behavioral Signals (Redrob Signals):**
| Signal | Weight Direction | Why |
|--------|-----------------|-----|
| search_appearance_30d | +++ | 5x gap between true ML and trap |
| saved_by_recruiters_30d | +++ | 4.3x gap |
| endorsements_received | ++ | 2.9x gap |
| connection_count | ++ | 2.2x gap |
| recruiter_response_rate | ++ | +44% gap |
| avg_response_time_hours | -- | -62% gap (lower is better) |
| open_to_work_flag | + | +104% more likely for true ML |
| interview_completion_rate | + | +27% gap |
| github_activity_score | + | +99% gap (if not -1) |
| profile_completeness_score | + | +32% gap |
| notice_period_days | -- | ≤30 = strong positive |
| verified_email/phone | + | Trust signal |

### 3.3 Trap Detection Phase (~2 seconds)
Apply these **trap classifiers** in priority order:

**Trap Type 1: Keyword Stuffer**
- Non-technical title AND
- 4+ AI/ML skills AND
- Zero foundational ML skills (PyTorch, TensorFlow, Deep Learning, NLP, Machine Learning) AND
- Skills are only LLM buzzwords (RAG, LangChain, Embeddings, Fine-tuning LLMs, Recommendation Systems)
- → **Apply -0.60 penalty to score**

**Trap Type 2: Template Summary**
- Summary contains "Lately I've been curious about how AI tools could augment my work" phrase
- → **Apply -0.30 penalty** (unless already caught by Type 1)

**Trap Type 3: Consulting-Only Career**
- ALL employers in career history are consulting/services companies (TCS, Infosys, Wipro, Accenture, Cognizant, etc.)
- → **Apply -0.25 penalty**

**Trap Type 4: Title-Chaser**
- 4+ jobs in last 8 years with average tenure <18 months
- → **Apply -0.15 penalty**

### 3.4 Honeypot Detection (~1 second)
Scan for impossible profiles:
- Years of experience > company age (check if current company age < experience)
- "Expert" in 5+ skills with 0 duration_months
- Candidate with contradictory data (e.g., current_title "Senior AI Engineer" but 0 AI skills AND 0 AI career history)
- → **Force score to 0.0** (automatic bottom)

### 3.5 Scoring Phase (~60 seconds)
**Composite Score = Weighted sum of signal scores × Trap penalties × Honeypot filter**

**Scoring formula**:

```
raw_score = 
  (title_relevance × 0.25) +
  (experience_fit × 0.12) +
  (product_exp × 0.10) +
  (ai_skills_depth × 0.15) +
  (career_relevance × 0.10) +
  (education_score × 0.03) +
  (behavioral_score × 0.15) +
  (location_fit × 0.05) +
  (availability_score × 0.05)

final_score = raw_score × trap_multiplier
```

**Where:**

**title_relevance (0.25)**: 6-level mapping:
| Title Category | Score | Examples |
|---------------|-------|----------|
| Core AI/ML | 1.00 | ML Engineer, AI Engineer, NLP Engineer, Data Scientist, Search Engineer, Recommendation Systems Engineer |
| ML-adjacent SWE | 0.85 | Senior Software Engineer (ML), Backend Engineer (with AI skills) |
| Software Engineer | 0.70 | Software Engineer, Full Stack Developer |
| Data/Cloud | 0.60 | Data Engineer, Data Analyst, Cloud Engineer, DevOps |
| Tech-adjacent | 0.40 | QA Engineer, Analytics Engineer |
| Non-technical | 0.05 | Everything else |

**experience_fit (0.12)**: Gaussian peak at 6-8 years:
| Years | Score |
|-------|-------|
| 5-9 | 1.00 |
| 4-4.9 or 9.1-12 | 0.80 |
| 3-3.9 or 12.1-14 | 0.50 |
| <3 or >14 | 0.20 |

**product_exp (0.10)**: 0.0 if consulting-only, 0.6 if mixed, 1.0 if all product companies

**ai_skills_depth (0.15)**:
- Has foundational ML (PyTorch, TF, NLP, DL, ML) → +0.4
- Has production AI tools (MLflow, Kubeflow, W&B, Docker, Kubernetes) → +0.2
- Has ranking/search/retrieval specific → +0.2
- Skills endorsed by other people → +0.1
- Skill assessment scores exist → +0.1
- Normalized by total skill count to prevent inflation gaming

**career_relevance (0.10)**:
- Has "ML Engineer", "Data Scientist", "AI" job title in history → +0.5
- Has search/ranking/recommendation keywords in descriptions → +0.3
- Has AI/ML project or product references → +0.2

**education_score (0.03)**: tier_1 = 1.0, tier_2 = 0.7, tier_3 = 0.4, tier_4 = 0.1, unknown = 0.3

**behavioral_score (0.15)**: Normalized composite of signals:
- `search_appearance_30d`: log-normalized (high = strong signal)
- `saved_by_recruiters_30d`: log-normalized  
- `recruiter_response_rate`: linear (higher = better)
- `avg_response_time_hours`: inverse (lower = better)
- `connection_count`: log-normalized
- `endorsements_received`: log-normalized
- `interview_completion_rate`: linear
- `open_to_work_flag`: binary boost
- `github_activity_score`: normalized (if > 0)

**location_fit (0.05)**: India = 1.0, US/Canada/UK = 0.5, other = 0.3

**availability_score (0.05)**: notice ≤30 = 1.0, ≤60 = 0.7, ≤90 = 0.4, >90 = 0.2

**trap_multiplier**: 
- No trap detected: 1.0
- Type 1 (keyword stuffer): 0.40
- Type 2 (template summary): 0.70 (if not Type 1)
- Type 3 (consulting-only): 0.75
- Type 4 (title-chaser): 0.85
- Multiple traps: multiplicative (but floor at 0.30)

### 3.6 Ranking + Reasoning Phase (~10 seconds)
1. Sort all 100K candidates by `final_score` descending
2. Take top 100
3. For each top-100 candidate, generate 1-2 sentence reasoning string
4. Assign ranks 1-100, ensure scores are non-increasing
5. Handle ties: break by candidate_id ascending

**Reasoning template** (anti-hallucination: only mention facts from the actual profile):
```
"[Current title] with {X} yrs; {AI skills count} AI/ML skills; 
{relevant career fact from history}; 
{signal strength: e.g., high recruiter engagement / strong response rate};
{concern if any: e.g., 90-day notice / consulting current - but has product history}"
```

Example reasoning:
- "ML Engineer with 6.4yr at Wysa; built ranking systems; strong recruiter engagement (saved 32x, 568 views/30d); 85% response rate, 90-day notice."
- "Senior Software Engineer (ML) with 4.1yr at Razorpay; recommendation system exp; tier_1 education, strong signals (high search appearance + endorsements)."

---

## 4. FILES STRUCTURE

```
├── rank.py                  # Main entry point (single command: python rank.py)
├── config.py                # All weights, thresholds, title mappings, skill ontologies
├── parser.py                # JSONL parser
├── features.py              # Feature extraction module
├── trap_detector.py         # Trap/honeypot classifiers
├── scoring.py               # Scoring engine
├── reasoning.py             # Reasoning string generator
├── output.py                # CSV writer + validator
├── utils.py                 # Shared utilities (log normalization, etc.)
├── requirements.txt         # Dependencies (stdlib only ideally)
├── submission_metadata.yaml # Team metadata
├── ARCHITECTURE.md          # This file
└── README.md                # Setup + reproduction instructions
```

**Dependencies strategy**: Minimize to zero or near-zero external deps. Goal: only stdlib (`json`, `csv`, `re`, `math`, `sys`). This guarantees:
- Works in any Python environment without `pip install`
- No version conflicts in sandboxed reproduction
- Fast cold-start

---

## 5. COMPUTE BUDGET (5 min = 300 sec)

| Phase | Est. Time | Notes |
|-------|-----------|-------|
| Parse 100K JSONL | ~6s | Pure stdlib json |
| Feature extraction (100K) | ~30s | Loop with dict lookups |
| Trap detection (100K) | ~5s | String matching + rules |
| Scoring (100K) | ~5s | Weighted sum arithmetic |
| Sort + top-100 extraction | ~2s | Python sorted() |
| Reasoning generation (100) | ~10s | Template filling |
| CSV output + validate | ~0.5s | |
| **Total** | **~58.5s** | **Well within 300s budget** |

No GPU, no network, no embedding models. The entire pipeline is CPU-based rule evaluation.

**Memory**: Loading 100K candidate dicts ~1GB. Peak memory ~2GB with intermediate feature arrays. Well within 16GB limit.

---

## 6. ANTI-TRAP STRATEGY (Critical for Winning)

### Why this wins:
The JD explicitly calls out the trap: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."* Most teams will:
1. Use embeddings → miss the trap → keyword-stuffed candidates rank high → honeypot rate >10% → **disqualified**
2. Use LLM API calls → violates network constraint → **disqualified**
3. Use simple keyword matching → rank non-technical candidates high → **low NDCG**

Our approach **explicitly detects and penalizes** the trap patterns while **boosting** the true gems.

### How we detect true gems that don't use AI buzzwords:
A candidate who built a recommendation system at a product company won't necessarily have "RAG" or "Pinecone" in their skills. But their career history descriptions WILL contain:
- "built recommendation engine for [product]"
- "designed search ranking pipeline"
- "implemented candidate-job matching system"
- "worked on information retrieval"

Our feature extraction catches these through career description keyword matching.

---

## 7. HONEYPOT PROTECTION

Honeypots (~80 candidates) have impossible profiles:
- **Time paradox**: e.g., "8 years at company founded 3 years ago"
- **Skill-fake**: "expert" in 10+ skills but 0 months duration for all
- **Experience vs. title mismatch**: Senior AI Engineer title but zero AI skills + zero AI career history

Our honeypot detector checks:
```
if candidate has "Senior AI Engineer" title AND 0 AI/ML skills AND no AI career history:
    score = 0.0 (guaranteed bottom)
```

This ensures honeypot rate in top 100 stays at 0%.

---

## 8. REASONING QUALITY

For Stage 4 manual review, reasonings must:
- Reference specific facts from profile (years, title, named skills, signal values)
- Connect to JD requirements
- Acknowledge honest concerns
- Be unique per candidate (not templated)
- Vary in tone by rank

We generate reasonings from actual extracted features (no hallucination risk since we only interpolate verified fields).

**Guard against hallucination**: The reasoning generator uses ONLY fields we've confirmed exist in the profile. No fabricated skills, employers, or experience.

---

## 9. IMPLEMENTATION ORDER

| Step | Task | Est. time |
|------|------|-----------|
| 1 | Create file structure + config.py with all weights/ontologies | ~1hr |
| 2 | Implement parser.py (JSONL reader) | ~30min |
| 3 | Implement features.py (extract all 40+ features) | ~2hr |
| 4 | Implement trap_detector.py (4 trap types + honeypots) | ~1hr |
| 5 | Implement scoring.py (weighted sum engine) | ~1hr |
| 6 | Implement reasoning.py (template generator) | ~1hr |
| 7 | Implement output.py (CSV writer + internal validator) | ~30min |
| 8 | Implement rank.py (orchestrator) | ~30min |
| 9 | Manual tune weights on sample 1000 candidates | ~2hr |
| 10 | Run on full 100K, validate output format | ~30min |
| 11 | Test with validate_submission.py | ~10min |
| 12 | Write README + submission_metadata.yaml | ~30min |

---

## 10. RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| Trap detection misses novel trap patterns | Broader keyword matching; any non-technical title with AI skills gets partial penalty |
| Honeypot in top 100 | Explicit honeypot detection loop; honeypots forced to score 0.0 |
| Score ties causing incorrect rank order | Tiebreak by candidate_id ascending per spec |
| Reasoning is too templated | 8 different reasoning templates with slot-filling; vary phrasing by score tier |
| Runtime near limit | Under 60s estimate; 300s budget gives 5x headroom |
| Memory spike | No model loading; only Python dicts (~1GB) |

---

## 11. WHY THIS WINS THE HACKATHON

1. **NDCG@10 (50%)**: Our careful trap filtering ensures the top 10 are genuinely relevant AI/ML engineers, not keyword stuffers. Behavioral signal weighting ensures they're actually available and engaged.

2. **NDCG@50 (30%)**: We capture the full breadth of true AI/ML talent while properly deprioritizing adjacent candidates.

3. **MAP (15%)**: Penalties for consulting-only careers and title-chasers ensure relevance precision across ranks.

4. **P@10 (5%)**: Bonus metric, same as NDCG@10 — we'll clean sweep.

5. **Honeypot rate**: Explicit detection → 0% in top 100 → pass Stage 3 filter.

6. **Reasoning quality**: Specific, honest, varied, connected to actual profile data → pass Stage 4.

7. **Code reproduction**: Pure stdlib, no external deps, single command → passes Stage 3 reproduction.

8. **Stage 5 interview**: We understand every weight and rule; no black-box model to defend.

9. **Compute efficiency**: ~60s runtime on a single CPU core → 5x under budget.

10. **No network calls**: Violates zero rules → no disqualification risk.

---

## 12. POST-SUBMISSION

- Provide GitHub repo with full code + README + requirements.txt
- Deploy sandbox on HuggingFace Spaces (Streamlit app with sample input)
- submission_metadata.yaml mirrors all portal metadata
- Ready for Stage 3 reproduction and Stage 5 interview

---

*This architecture prioritizes correctness, interpretability, and reproducibility over model complexity — exactly what the evaluation pipeline rewards.*
