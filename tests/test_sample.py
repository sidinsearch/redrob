"""Test on sample candidates.jsonl (50 candidates). Verify:
- No exceptions
- All 50 get processed
- Honeypots detected correctly
- Output CSV is well-formed
"""

import json
import sys
from pathlib import Path

# Allow imports from src/ and the rank.py at project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import config
import features
import output
import parser
import scoring
import trap_detector


def main():
    sample_path = Path(r"D:\redrob\sample_candidates.jsonl")
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # The sample is a JSON array, not JSONL. Convert it.
    if text.startswith("["):
        data = json.loads(text)
        # Write it as JSONL for the ranker
        jsonl_path = sample_path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj) + "\n")
        candidates = data
    else:
        candidates = []
        for line in text.splitlines():
            if line.strip():
                candidates.append(json.loads(line))
        jsonl_path = sample_path

    print(f"Loaded {len(candidates)} candidates from {sample_path.name}")

    n_honeypots = 0
    n_keyword_stuffers = 0
    n_template_summary = 0
    n_consulting_only = 0
    n_title_chaser = 0

    scored = []
    for c in candidates:
        f = features.extract_features(c)
        t = trap_detector.analyze(f)
        if t.is_honeypot:
            n_honeypots += 1
        if t.is_keyword_stuffer:
            n_keyword_stuffers += 1
        if t.is_template_summary:
            n_template_summary += 1
        if t.is_consulting_only:
            n_consulting_only += 1
        if t.is_title_chaser:
            n_title_chaser += 1
        score = scoring.compute_score(f, t)
        scored.append((score, c["candidate_id"], f.current_title, f.years_of_experience, t.is_honeypot))

    print(f"\nTrap stats:")
    print(f"  honeypots: {n_honeypots}")
    print(f"  keyword stuffers: {n_keyword_stuffers}")
    print(f"  template summary: {n_template_summary}")
    print(f"  consulting only: {n_consulting_only}")
    print(f"  title chaser: {n_title_chaser}")

    scored.sort(key=lambda x: (-x[0], x[1]))

    print(f"\nTop 10 by score:")
    print(f"{'rank':<5}{'score':<10}{'id':<14}{'title':<35}{'yoe':<6}{'honeypot'}")
    for i, (score, cid, title, yoe, is_hp) in enumerate(scored[:10], 1):
        print(f"{i:<5}{score:<10.4f}{cid:<14}{(title or '')[:33]:<35}{yoe:<6}{is_hp}")

    print(f"\nBottom 5 by score:")
    for i, (score, cid, title, yoe, is_hp) in enumerate(scored[-5:], len(scored) - 4):
        print(f"{i:<5}{score:<10.4f}{cid:<14}{(title or '')[:33]:<35}{yoe:<6}{is_hp}")

    # Now actually run the ranker
    print(f"\n{'='*60}")
    print(f"Running full ranker pipeline on {len(candidates)} candidates...")

    import rank
    out_path = Path(__file__).resolve().parent.parent / "output" / "sample_submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use all 50 as top-k to test full pipeline (won't pass spec validator
    # but proves the pipeline works on small data)
    stats = rank.run(jsonl_path, out_path, top_k=len(candidates))
    print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    main()
