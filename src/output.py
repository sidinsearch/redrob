"""output.py — CSV writer + format validation.

Writes submission CSV per the spec:
- Row 1: header `candidate_id,rank,score,reasoning`
- Rows 2-101: exactly 100 data rows
- candidate_id format: CAND_XXXXXXX (7 digits)
- rank: integer 1-100, used exactly once
- score: float, non-increasing by rank, ties broken by candidate_id ascending

Includes a built-in validator that mirrors validate_submission.py.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple

import config

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def write_submission(rows: List[Tuple[str, int, float, str]], path: str | Path) -> None:
    """Write ranked rows to CSV.

    rows: list of (candidate_id, rank, score, reasoning) tuples.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(REQUIRED_HEADER)
        for cid, rank, score, reasoning in rows:
            # Normalize reasoning: strip newlines, cap length
            r = (reasoning or "").replace("\n", " ").replace("\r", " ").strip()
            if len(r) > config.MAX_REASONING_LEN:
                r = r[: config.MAX_REASONING_LEN - 1].rstrip() + "…"
            # Use 6 decimal places to avoid score-tie collisions at 4dp
            writer.writerow([cid, int(rank), f"{float(score):.6f}", r])


def validate(path: str | Path) -> List[str]:
    """Self-validate the output CSV. Returns list of error strings (empty if valid)."""
    errors = []
    path = Path(path)

    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("Empty file")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(f"Header must be {REQUIRED_HEADER}, got {header}")

            data_rows = []
            for row in reader:
                if any(cell.strip() for cell in row):
                    data_rows.append(row)
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded.")
        return errors

    n = len(data_rows)
    if n != config.TOP_K:
        errors.append(f"Expected {config.TOP_K} data rows, got {n}")

    seen_ids = set()
    seen_ranks = set()
    by_rank = []

    for i, cells in enumerate(data_rows):
        row_num = 2 + i
        if len(cells) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(cells)}")
            continue
        cid, rank_s, score_s, _ = cells
        cid = cid.strip()
        rank_s = rank_s.strip()
        score_s = score_s.strip()

        if not cid or not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: bad candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank must be 1-100")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be an integer")
            rank = None

        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be a float")
            score = None

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"score not non-increasing: rank {r1} ({s1}) < rank {r2} ({s2})")
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(f"tie-break violated at ranks {r1}/{r2}: {c1} > {c2}")

    return errors
