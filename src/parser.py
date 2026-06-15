"""parser.py — JSONL streaming reader with byte-offset index.

Why a byte-offset index:
- candidates.jsonl is 487 MB. Loading the whole thing into memory is wasteful.
- The challenge asks for top-100 candidates. We rank everything but only
  need to load 100 of them at the end to generate reasoning strings.
- A byte-offset index lets us seek straight to those 100 lines for
  sub-second reasoning generation.

ponytail: this module does NOT load candidates into memory eagerly. We
process in streaming fashion and only seek-to-line when needed.

Memory: ~8 bytes per offset * 100K = ~800 KB. Trivial.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple


def _iter_jsonl(path: str | os.PathLike) -> Iterator[Tuple[int, int, dict]]:
    """Yield (byte_offset, line_number, parsed_dict) tuples.

    byte_offset is the start of the line in the file.
    line_number is 0-indexed (i.e. line_number 0 is the first candidate).
    """
    with open(path, "rb") as f:
        offset = 0
        line_number = 0
        for raw in f:
            raw = raw.rstrip(b"\n")
            if raw:
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    # Skip malformed lines; report via offset
                    pass
                else:
                    yield offset, line_number, obj
            offset += len(raw) + 1  # +1 for the newline
            line_number += 1


def build_offset_index(path: str | os.PathLike) -> Tuple[List[int], List[str]]:
    """Walk the JSONL once to build a (offset, candidate_id) index.

    Returns:
        offsets: list of byte offsets, one per candidate, in file order.
        ids: parallel list of candidate_id strings.
    """
    offsets: List[int] = []
    ids: List[str] = []
    for offset, _line_num, obj in _iter_jsonl(path):
        offsets.append(offset)
        ids.append(obj.get("candidate_id", ""))
    return offsets, ids


def load_all(path: str | os.PathLike) -> List[dict]:
    """Load every candidate into memory. Use only for small inputs (≤5K).

    For the full 100K pool, use iter_all() with the streaming pipeline.
    """
    return [obj for _, _, obj in _iter_jsonl(path)]


def iter_all(path: str | os.PathLike) -> Iterator[dict]:
    """Yield candidates one at a time, no in-memory accumulation."""
    for _, _, obj in _iter_jsonl(path):
        yield obj


def seek_to_lines(path: str | os.PathLike, offsets: List[int]) -> List[dict]:
    """Given a list of byte offsets, seek-and-parse just those lines.

    Used at the end of ranking to load only the top-100 candidates for
    reasoning. With 100 seeks over a 487 MB file, this is ~100 ms.
    """
    results: List[dict] = []
    if not offsets:
        return results
    with open(path, "rb") as f:
        for offset in offsets:
            f.seek(offset)
            line = f.readline()
            try:
                results.append(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                results.append({})
    return results


def count_lines(path: str | os.PathLike) -> int:
    """Count non-empty lines in a JSONL file. Fast: just newline counting."""
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def path_size_mb(path: str | os.PathLike) -> float:
    return Path(path).stat().st_size / (1024 * 1024)
