"""redrob-ranker — Redrob hackathon candidate ranking system.

A stdlib-only, CPU-only, no-network ranker that:
1. Streams 100K candidates from JSONL
2. Extracts 40+ features per candidate
3. Detects honeypots + 4 trap types
4. Computes a composite score
5. Ranks, generates per-candidate reasoning
6. Writes submission.csv

Public API: `rank.main()`
"""

__version__ = "1.0.0"
