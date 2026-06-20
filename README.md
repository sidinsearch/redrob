# Redrob Candidate Ranker

> **100,000 candidates → top 100 best matches** for the Senior AI Engineer —
> Founding Team JD. Built for the [Redrob AI Challenge](https://hack2skill.com/event/india_runs/).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-40%2F40_passing-brightgreen.svg)](tests/)
[![Stdlib ranker](https://img.shields.io/badge/ranker-stdlib_only-blue.svg)](src/)
[![CPU only](https://img.shields.io/badge/compute-CPU_only-orange.svg)](docs/ARCHITECTURE.md)
[![No network](https://img.shields.io/badge/network-offline-lightgrey.svg)](docs/ARCHITECTURE.md)
[![0% honeypots](https://img.shields.io/badge/top100_honeypots-0%25-success.svg)](docs/METHODOLOGY.md)

## What this does

Given a 100K-candidate pool (`candidates.jsonl`) and the released JD for a
Senior AI Engineer — Founding Team role, this ranker produces a top-100 CSV
with `candidate_id`, `rank`, `score`, `reasoning`.

The right answer isn't *"find candidates whose skills section contains the
most AI keywords."* That's a trap explicitly built into the dataset. Our
ranker reads the gap between what the JD says and what the JD means.

**One-command reproduce:**

```bash
python rank.py --candidates /path/to/candidates.jsonl --out ./output/submission.csv
```

~35 s on a 16 GB CPU-only machine. Validates the output CSV before exiting.

## Quick start

```bash
git clone https://github.com/sidinsearch/redrob.git
cd redrob

# Sandbox needs only streamlit + pandas + plotly. The ranker itself is stdlib.
pip install -r requirements.txt

# Headless: produce submission.csv
python rank.py --candidates /path/to/candidates.jsonl --out ./output/submission.csv

# Interactive: open the Streamlit sandbox in your browser
streamlit run app.py
```

Or skip the install and go straight to [Docker](#docker-deploy) — one build,
one command, browser opens.

## What's in the top-100

The current submission's top-5 (100K candidate pool, 0 honeypots in top-100):

| Rank | ID | Score | Title | Company | YoE | Must-haves |
|-----:|-----|------:|-------|---------|----:|:----------:|
| 1 | CAND_0055905 | 0.7965 | Senior ML Engineer | Flipkart | 8.1 | 3/4 |
| 2 | CAND_0011687 | 0.7650 | Senior NLP Engineer | Niramai | 7.8 | 3/4 |
| 3 | CAND_0046064 | 0.7649 | Senior NLP Engineer | Salesforce | 8.9 | 3/4 |
| 4 | CAND_0064326 | 0.7546 | Search Engineer | Sarvam AI | 7.6 | 3/4 |
| 5 | CAND_0018499 | 0.7511 | Senior ML Engineer | Zomato | 7.2 | 3/4 |

## How the score is built

Per the audit spec, `final_score = (fit_score / 100) × availability_multiplier`.

```
                      candidates.jsonl (100K)
                              │
                              ▼
      ┌────────────────────────────────────────────────────────────────┐
      │  1. PARSE  JSONL streaming reader                              │
      │                                                                │
      └────────────────────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────────────────────┐
      │  2. FEATURES  70+ signals per candidate                        │
      │      title · YoE · AI-skill depth                              │
      │      product-company exp · education · location                │
      │                                                                │
      └────────────────────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────────────────────┐
      │  3. MUST-HAVES  evidence-based detector                        │
      │      4 must-haves (career_history sentences)                   │
      │      5 nice-to-haves (FAISS, PyTorch, …)                       │
      │      + hard disqualifiers (research/consulting/LangChain)      │
      │                                                                │
      └────────────────────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────────────────────┐
      │  4. FIT-SCORE  0–100, gated on must-haves met (0–4)            │
      │      4/4 → [80,100]   3/4 → [55,75]                            │
      │      2/4 → [30, 50]   1/4 → [10,25]   0/4 → [0,10]             │
      │                                                                │
      └────────────────────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────────────────────┐
      │  5. TRAPS  4 multiplicative + 12 honeypot flags                │
      │      stuffer ×0.40  │  template ×0.70                          │
      │      consulting ×0.75  │  chaser ×0.85                         │
      │      → honeypots excluded from top-K (architectural)           │
      │                                                                │
      └────────────────────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────────────────────┐
      │  6. AVAILABILITY  additive, clipped to [0.1, 1.0]              │
      │      open_to_work 40%  │  notice_period 25%                    │
      │      response_rate 20%  │  recency 15%                         │
      └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    final_score = (fit_score / 100) × availability_multiplier
                              │
                              ▼
      sort desc by (score, candidate_id) → top-100
                              │
                              ▼
      8 reasoning templates × 4 rank tiers  (≤300 chars)
                              │
                              ▼
      output/submission.csv      output/submission.honeypots.csv
```

**Honeypots never reach the top-K.** `rank.py` and `app.py` both call
`continue` *before* `topk.offer()` for any candidate whose `TrapResult.is_honeypot`
is true. This is an architectural guarantee, not a score threshold — even a
honeypot with a perfect fit_score cannot appear in the top-100.

## Traps and honeypots

| Trap | Pattern | Multiplier |
|------|---------|------------|
| **Honeypot** | Career-timeline impossible, expert-with-zero-duration, AI title with 0 AI skills, achievement inflation, employment overlap, technology-age anomaly (≥2 anachronistic skills), title-skills mismatch, etc. (12 detectors) | **excluded** from top-K |
| **Keyword stuffer** | Non-technical title + 4+ AI skills + zero foundational ML | × 0.40 |
| **Template summary** | Canned "curious about AI tools" phrasing in summary | × 0.70 |
| **Consulting only** | All employers are TCS / Infosys / Wipro / Accenture / Cognizant | × 0.75 |
| **Title chaser** | 4+ jobs in last 8 yr with avg tenure <18 months | × 0.85 |

Multiple traps multiply; floor at `0.30`. Honeypots are an *exclusion*, not a
multiplier — they are pulled out before ranking.

**Detector tightening** (commit `77d8ca8`):
- `HONEYPOT_YOE_BUFFER_YEARS` widened 5 → 8 to absorb PhD / parental / military gaps
- `NLP_KEYWORDS` expanded 9 → 41 terms (catches RAG, BGE, GPT, etc.)
- `technology_age_anomaly` now requires ≥2 anachronistic skills (was 1)
- 6 honeypots that were leaking into the top-100 in earlier versions are now caught

## Project structure

```
redrob/
├── rank.py                       # Main entry — `python rank.py --candidates ...`
├── app.py                        # Streamlit sandbox — 6 tabs, weight tuning, score breakdown
├── requirements.txt              # streamlit, pandas, plotly (sandbox only)
├── sample_candidates.jsonl       # 50-candidate sample bundled for the sandbox
├── submission_metadata.yaml      # Stage-3 portal metadata
├── Dockerfile                    # Multi-mode container (CLI + sandbox)
├── docker-entrypoint.sh          # Defaults to streamlit on :8501; passthrough otherwise
├── .streamlit/config.toml        # maxUploadSize=500, headless, no telemetry
├── src/
│   ├── __init__.py
│   ├── config.py                     # weights, MUST_HAVE_PATTERNS, NLP_KEYWORDS,
│   ├──                               # AVAILABILITY_WEIGHTS, FIT_TIER_BASELINES,
│   ├──                               # HONEYPOT_YOE_BUFFER_YEARS, etc.
│   ├── parser.py                     # JSONL streaming reader
│   ├── features.py                   # 70+ signals per candidate (76 fields)
│   ├── must_haves.py                 # 4 must-haves + 5 nice-to-haves + hard disqualifiers
│   ├── trap_detector.py              # 4 trap types + honeypot composite (12 flags)
│   ├── scoring.py                    # compute_final_score() → fit, availability, final
│   ├── reasoning.py                  # 8 templates × 4 rank tiers, ≤300 chars
│   ├── output.py                     # CSV writer + format validator
│   ├── ranking_pipeline.py           # Reusable pipeline used by app.py
│   └── utils.py                      # Shared helpers (log normalization, etc.)
├── tests/                        # 40/40 passing
│   ├── test_traps.py                 # 9
│   ├── test_output.py                # 6
│   ├── test_honeypots.py             # 3
│   ├── test_honeypots_extended.py    # 13
│   ├── test_honeypots_excluded.py    # 2 — regression: honeypots never in top-K
│   ├── test_must_haves.py            # 10
│   └── test_sample.py
├── docs/
│   ├── ARCHITECTURE.md               # Full design doc (weights, data flow, compute budget)
│   ├── METHODOLOGY.md                # How to defend the design at Stage 5
│   └── EVALUATION.md                 # Local evaluation strategy
└── output/                       # submission.csv + submission.honeypots.csv (gitignored)
```

## Performance

| Metric | Value |
|--------|-------|
| Runtime on 100K pool | ~35 s (8.5× under the 5-min budget) |
| Memory | ~1.5 GB peak (well under 16 GB) |
| Network | 0 calls (offline-only by design) |
| External deps (ranker) | 0 (stdlib only) |
| External deps (sandbox) | 3 (streamlit, pandas, plotly) |
| Honeypot rate in top-100 | **0%** (target: 0%; disqualification threshold: >10%) |
| Tests | 40/40 passing in <0.2 s |

## Streamlit sandbox

`app.py` is a 6-tab interactive harness over the same ranker:

1. **Top Candidates** — sortable dataframe, CSV + JSON download
2. **Trap Analysis** — pie + bar of trap distribution
3. **Score Breakdown** — per-candidate must-have evidence, fit × availability
4. **Candidate Drill-down** — profile, trap status, skills, career, reasoning
5. **Fairness Audit** — country, education, YoE distribution of top-100
6. **Excluded Honeypots** — full honeypot details + per-detector counts

The sidebar exposes live weight tuning (must sum to 1.0) — drag a slider and the
top-100 re-ranks in <1 s. Upload supports up to 500 MB per Streamlit's
`.streamlit/config.toml`.

## Docker deploy

A single image handles every mode: CLI ranking or Streamlit sandbox.

```bash
docker build -t redrob-ranker .
```

> **Always rebuild after `git pull`.** The image is a snapshot of the source
> at build time; `git pull` does not update a running or already-built
> image. If you skip this, the container will keep running the old code.

**Streamlit sandbox (default — opens in browser on http://localhost:8501):**

```bash
docker run --rm -p 8501:8501 redrob-ranker
```

**CLI ranking (headless, one-shot — writes the submission CSV and exits):**

```bash
docker run --rm \
  -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
  -v /path/to/output:/output \
  redrob-ranker \
  python rank.py --candidates /data/candidates.jsonl --out /output/submission.csv
```

**Dispatch rule:** the entrypoint runs Streamlit on port 8501 when called
with no args, and otherwise passes the command through verbatim. See
`docker-entrypoint.sh` (~15 lines).

## Why rule-based (over neural embeddings)?

The JD itself gives the answer: *"We've tried BM25 + rule-based, working
but not great."* That sentence is the rubric. Rule-based scoring is the
baseline the org knows. Improvements are visible against it. Neural
embeddings per-candidate would violate the 5-min CPU budget and require
pre-computed artifacts (also out-of-budget at reproduction time). Pure
embedding-based approaches also tend to rank the keyword stuffers high
because their skill list is "embedding-similar" to the JD even when their
career is nonsense.

Our approach is **defensible at Stage 5**:
- Every weight is a single constant in `src/config.py` — easy to defend.
- Every trap rule is a 5-line detector — easy to walk through.
- Every reason is grounded in a real profile field — zero hallucination.
- Pure stdlib — no model version, no embedding drift, no surprises at reproduction.

## Local evaluation strategy

We don't have the hidden ground truth. The best proxies:

1. **Honeypot rate in top-100 = 0%** — survives Stage 3.
2. **All top-100 candidates have at least one product-company employer** — the
   JD explicitly says "applied ML at product companies".
3. **Distribution sanity check** — top-10 by score vs random-10 should have a
   meaningful gap in skills/title signals.
4. **Reasoning quality** — every fact in the reasoning must come from the
   actual profile (manual review).

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the full methodology.

## Tests

```bash
python -m pytest tests/ -q
# 40 passed in 0.11s
```

Coverage:
- **9** trap detection tests
- **6** output / validator tests
- **3** full-data honeypot verification
- **13** honeypot detector unit tests
- **2** regression tests: honeypots never in top-K (both `rank.py` and `app.py`)
- **10** must-have evidence detection tests

## License

MIT.
