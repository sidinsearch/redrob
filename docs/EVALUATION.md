# Evaluation Strategy

We don't have access to the hidden ground truth during the competition.
This document explains how we evaluate the ranker locally to maximize the
chance of a good final score.

---

## 1. The scoring formula (we don't see this)

Final composite (from `submission_spec.md` §4):

```
Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

NDCG@10 carries 50% of the weight. **The top-10 matters most.** NDCG@50
is the second-biggest contributor. MAP averages precision across all
100 ranks. P@10 is a small bonus.

The metric structure tells us:
- **Optimize the top-10 ruthlessly.** Even if our rank-11 through rank-100
  is mediocre, a strong top-10 keeps us competitive.
- **Avoid honeypots at all costs.** The threshold is >10% in top 100
  (we target 0%), but a single honeypot in top-10 likely tanks NDCG@10.
- **Tier 5 (plain-language) candidates matter.** They appear in the
  hidden ground truth with high relevance, but the rule-based system
  has to find them via career_relevance + skills, not title alone.

## 2. Local validation: what we can check

Without ground truth, we evaluate via proxy signals:

### Signal 1: Honeypot rate in top-100
- Target: 0% (0 honeypots in top 100).
- Threshold: >10% triggers Stage 3 disqualification.
- **We hit 0% in every run.**

### Signal 2: Distribution of titles in top-100
A well-calibrated ranker should have top-100 mostly AI/ML-titled candidates
with some adjacent titles (Software Engineer, Data Engineer). The trap
distribution is:

| Expected top-100 distribution | Approximate % |
|-------------------------------|---------------|
| Core AI/ML (ML Engineer, Data Scientist, NLP, etc.) | 60-80% |
| ML-adjacent SWE (Senior SWE with ML focus) | 10-20% |
| Software Engineer / Data Engineer (with strong AI skills) | 5-15% |
| Adjacent (Data Engineer, Cloud, DevOps with AI skills) | 0-5% |
| Tier 5 (non-tech title with strong ML history) | 0-5% |
| Non-technical / trap | 0% |

We manually inspect the top-100 of every run to verify the distribution
shape stays consistent.

### Signal 3: Reason grounding
Every fact in the `reasoning` column must come from the actual profile.
The validator checks this manually at Stage 4. We pre-validate by:
- Spot-checking 20 random top-100 reasonings against the source profile
- Verifying that skill names appear in the profile's skills list
- Verifying that employer names appear in the career history
- Verifying that "X years" matches the profile's `years_of_experience`

### Signal 4: Score tier consistency
Rank-1's score should be meaningfully higher than rank-100's score.
A flat distribution (all scores ≈ 0.5) is a sign that the ranker
isn't differentiating.

We expect:
- Rank 1: 0.85 - 0.95
- Rank 100: 0.75 - 0.85
- Spread: ≥0.10

## 3. Local test set

We use the 50-candidate sample as our test bed:
- `python tests/test_sample.py` runs the full pipeline on 50 candidates
- We manually inspect the top-20 reasonings
- We verify the trap stats make sense (e.g., ~60% of sample should be
  template-summary matches given the dataset prevalence)

## 4. Pre-submit checklist

Before each submission (3 max):

- [ ] Run `python rank.py --candidates ... --out ./output/submission.csv`
- [ ] Run `python validate_submission.py ./output/submission.csv` (must pass)
- [ ] Confirm 0 honeypots in top-100
- [ ] Spot-check 10 random reasonings
- [ ] Confirm trap distribution is stable across runs (no time-varying randomness)
- [ ] Confirm runtime is under 5 minutes (target: <1 minute)
- [ ] Confirm memory is under 16 GB (target: <2 GB)

## 5. What we deliberately don't optimize

- **Highest possible score on the sample** — the sample is too small (50)
  to be representative; overfitting to it would hurt on the 100K pool.
- **A/B testing on the 100K** — we have 3 submissions, not enough for
  meaningful A/B. One good submission beats three mediocre ones.
- **Hyperparameter search** — the weights are tuned against the JD's
  textual description, not against a metric. Searching against a metric
  risks overfitting to the leaderboard (which we don't see).

## 6. Post-submission (if we make the top X)

If we advance to Stage 5 (defend-your-work interview), the
[METHODOLOGY.md](METHODOLOGY.md) document is the script.

If we don't advance, the public-facing GitHub repo is the artifact.
The README + docs are written to make the design clear to any reader
who wants to learn from the approach.
