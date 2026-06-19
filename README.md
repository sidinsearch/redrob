# Redrob Candidate Ranker

> **100,000 candidates → top 100 best matches** for the Senior AI Engineer — Founding Team JD.
> Built for the [Redrob AI Challenge](https://hack2skill.com/event/india_runs/).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Stdlib only](https://img.shields.io/badge/dependencies-stdlib_only-green.svg)](requirements.txt)
[![CPU only](https://img.shields.io/badge/compute-CPU_only-orange.svg)](docs/ARCHITECTURE.md)
[![No network](https://img.shields.io/badge/network-offline-blue.svg)](docs/ARCHITECTURE.md)

## What this does

Given a 100K-candidate pool (`candidates.jsonl`) and the released JD for a Senior AI
Engineer — Founding Team role, this ranker produces a top-100 CSV with:
- `candidate_id`, `rank`, `score`, `reasoning`

**The right answer isn't "find candidates whose skills section contains the most
AI keywords."** That's a trap explicitly built into the dataset. Our ranker reads
the gap between what the JD says and what the JD means.

## Quick start

```bash
# Clone the repo
git clone https://github.com/sidinsearch/redrob-ranker
cd redrob-ranker

# Install ONLY Streamlit (for the sandbox app). The ranker itself is stdlib-only.
pip install -r requirements.txt

# Run the ranker on the full candidate pool
python rank.py --candidates /path/to/candidates.jsonl --out ./output/submission.csv

# Or, try the interactive sandbox
streamlit run app.py
```

Or skip the install and go straight to [Docker](#docker-deploy) — one command,
browser opens, done.

## Docker deploy

A single image handles every mode: CLI ranking or Streamlit sandbox. Rebuild
whenever you `git pull` new code, then pick your command on `docker run`.

```bash
# (Re)build the image — this picks up the latest code from the working tree
docker build -t redrob-ranker .
```

> **Always rebuild after `git pull`.** The image is a snapshot of the source
> at build time; `git pull` does not update a running or already-built
> image. If you skip this step, the container will keep running the old
> code (e.g., the top-100 will be in JSONL order, not ranked order).

**Streamlit sandbox (default — opens in browser on http://localhost:8501):**

```bash
# Bare launch — use the upload widget in the UI
docker run --rm -p 8501:8501 redrob-ranker

# Or mount your own data so it's pre-loaded
docker run --rm -p 8501:8501 \
  -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
  redrob-ranker
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

## Performance

| Metric | Value |
|--------|-------|
| Runtime on 100K pool | ~35 seconds (5x under the 5-min budget) |
| Memory | ~1.5 GB peak (well under 16 GB) |
| Network | 0 calls (offline-only by design) |
| External dependencies | 0 (ranker), 1 (Streamlit for sandbox) |
| Honeypot rate in top-100 | 0% (target: 0%, disqualification threshold: >10%) |

## Architecture

The ranker is a **rule-based weighted scoring system with explicit trap detection**.

```
candidates.jsonl (100K)
       │
       ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────────┐
│  Parser      │  │  Features      │  │  Trap Detector     │
│  JSONL       │→ │  40+ signals   │→ │  4 traps +         │→ score
│  streaming   │  │  per candidate │  │  3 honeypot types  │
└──────────────┘  └────────────────┘  └────────────────────┘
                                                   │
                                                   ▼
                                        ┌────────────────────┐
                                        │  Scoring Engine    │
                                        │  weighted sum ×    │
                                        │  trap multiplier   │
                                        └────────┬───────────┘
                                                 │
                                                 ▼
                                        ┌────────────────────┐
                                        │  Top-K + Reasoning │
                                        │  8 templates ×     │
                                        │  4 rank tiers      │
                                        └────────┬───────────┘
                                                 │
                                                 ▼
                                        submission.csv
                                        (100 ranked candidates)
```

For the full design doc — including every weight, threshold, and rationale — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Traps detected

The dataset contains four explicit trap patterns. Each gets a multiplicative
score penalty. Honeypots are forced to the bottom (0% in top-100 is the target).

| Trap | Pattern | Multiplier |
|------|---------|-----------|
| **Honeypot** | Impossible profile (YoE > career span, "expert" in 5+ skills with 0mo use, AI title with 0 AI skills) | `0.0` (forced bottom) |
| **Keyword stuffer** | Non-technical title + 4+ AI skills + zero foundational ML | `× 0.40` |
| **Template summary** | Summary contains the canned "curious about AI tools" phrase | `× 0.70` |
| **Consulting only** | All employers are TCS / Infosys / Wipro / Accenture / etc. | `× 0.75` |
| **Title chaser** | 4+ jobs in last 8 years with avg tenure <18 months | `× 0.85` |

Multiple traps multiply; floor at `0.30`.

## Project structure

```
redrob-ranker/
├── rank.py                  # Main entry point — `python rank.py --candidates ... --out ...`
├── app.py                   # Streamlit sandbox — `streamlit run app.py`
├── src/
│   ├── __init__.py
│   ├── config.py            # All weights, ontologies, thresholds (single source of truth)
│   ├── parser.py            # JSONL streaming reader with byte-offset index
│   ├── features.py          # Per-candidate feature extraction
│   ├── trap_detector.py     # 4 trap types + 3 honeypot patterns
│   ├── scoring.py           # Weighted composite + trap multiplier
│   ├── reasoning.py         # 8 templates × 4 rank tiers, no hallucination
│   ├── output.py            # CSV writer + format validator
│   ├── ranking_pipeline.py  # Reusable pipeline for the Streamlit app
│   └── utils.py             # Shared helpers (log normalization, etc.)
├── tests/
│   ├── test_sample.py       # Test on the 50-candidate sample
│   ├── test_traps.py        # Unit tests for trap detectors
│   └── test_honeypots.py    # Unit tests for honeypot detection
├── sample_candidates.jsonl  # 50-candidate sample, bundled for the Streamlit sandbox
├── docs/
│   ├── ARCHITECTURE.md      # Full design doc (weights, data flow, compute budget)
│   ├── METHODOLOGY.md       # How to defend the design at Stage 5
│   └── EVALUATION.md        # Local evaluation strategy
├── output/                  # Generated submission.csv (gitignored)
├── requirements.txt         # streamlit, pandas, plotly (the ranker is stdlib-only)
├── submission_metadata.yaml # For portal upload
├── Dockerfile               # Multi-mode container (CLI + sandbox)
├── docker-entrypoint.sh     # Defaults to streamlit; passthrough otherwise
├── .streamlit/config.toml   # maxUploadSize=500, headless, no telemetry
├── .dockerignore
├── .gitignore               # Excludes AI agentic files, candidate data, etc.
└── README.md                # This file
```

## Reproducing the submission

The single command that produces the submission CSV from the candidates file:

```bash
python rank.py --candidates /path/to/candidates.jsonl --out ./output/submission.csv
```

This completes in ~35 seconds on a 16 GB CPU-only machine with no network access.
Validates the output before exiting.

## Why this approach (over neural embeddings)?

The JD itself gives the answer: *"We've tried BM25 + rule-based, working but not great."*

That sentence is the rubric. Rule-based scoring is the baseline the org knows.
Improvements are visible against it. Neural embeddings per-candidate would
violate the 5-minute CPU budget and require pre-computed artifacts (also
out-of-budget at reproduction time). Pure embedding-based approaches also tend
to rank the keyword stuffers high because their skill list is "embedding-similar"
to the JD even when their career is nonsense.

Our approach is **defensible at Stage 5**:
- Every weight is a single constant in `src/config.py` — easy to defend.
- Every trap rule is a 5-line detector — easy to walk through.
- Every reason is grounded in a real profile field — zero hallucination.
- Pure stdlib — no model version, no embedding drift, no surprises at reproduction.

## Local evaluation strategy

We don't have the hidden ground truth. The best proxies:

1. **Honeypot rate in top-100 = 0%** — survives Stage 3.
2. **All top-100 candidates have at least one product-company employer** — the JD explicitly says "applied ML at product companies".
3. **Distribution sanity check** — top-10 by score vs random-10 should have a meaningful gap in skills/title signals.
4. **Reasoning quality** — every fact in the reasoning must come from the actual profile (manual review).

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the full evaluation methodology.

## License

MIT.
