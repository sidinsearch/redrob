"""test_output.py — Unit tests for CSV output and validator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
import output


def test_write_and_validate_round_trip(tmp_path=None):
    """Write a valid 100-row submission, validate, ensure no errors."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    for i in range(1, 101):
        cid = f"CAND_{i:07d}"
        rank = i
        score = 1.0 - (i - 1) * 0.001  # Strictly decreasing
        reasoning = f"Rank {i} candidate, ML engineer at Company{i}, 5 years experience."
        rows.append((cid, rank, score, reasoning))

    output.write_submission(rows, path)
    errors = output.validate(path)
    assert not errors, f"unexpected errors: {errors}"


def test_validator_catches_duplicate_ranks():
    """Validator must catch duplicate rank values."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    for i in range(1, 101):
        cid = f"CAND_{i:07d}"
        rank = 1 if i == 50 else i  # Duplicate rank 1
        score = 1.0 - (i - 1) * 0.001
        rows.append((cid, rank, score, "X"))
    output.write_submission(rows, path)
    errors = output.validate(path)
    assert any("duplicate" in e.lower() for e in errors), f"expected duplicate rank error: {errors}"


def test_validator_catches_score_inversion():
    """Validator must catch non-monotonic scores."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    for i in range(1, 101):
        cid = f"CAND_{i:07d}"
        rank = i
        # Invert score: lower rank gets LOWER score
        score = 0.5 + (i - 1) * 0.001
        rows.append((cid, rank, score, "X"))
    output.write_submission(rows, path)
    errors = output.validate(path)
    assert any("non-increasing" in e.lower() or "not non-increasing" in e.lower() for e in errors), f"expected score inversion error: {errors}"


def test_validator_catches_bad_candidate_id():
    """Validator must catch malformed candidate_id."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    for i in range(1, 101):
        cid = "INVALID_ID" if i == 1 else f"CAND_{i:07d}"
        rank = i
        score = 1.0 - (i - 1) * 0.001
        rows.append((cid, rank, score, "X"))
    output.write_submission(rows, path)
    errors = output.validate(path)
    assert any("bad candidate_id" in e.lower() for e in errors), f"expected bad id error: {errors}"


def test_validator_catches_wrong_row_count():
    """Validator must catch 99 or 101 rows."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    for i in range(1, 100):  # Only 99 rows
        cid = f"CAND_{i:07d}"
        rank = i
        score = 1.0 - (i - 1) * 0.001
        rows.append((cid, rank, score, "X"))
    output.write_submission(rows, path)
    errors = output.validate(path)
    assert any("expected 100" in e.lower() or "data rows" in e.lower() for e in errors), f"expected row count error: {errors}"


def test_tie_break_cid_ascending():
    """Validator must enforce cid ascending on equal scores."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        path = f.name

    rows = []
    # Two candidates with same score, but cids are not in ascending order
    rows.append(("CAND_0000009", 1, 0.5, "Higher cid, but rank 1"))  # bad
    rows.append(("CAND_0000001", 2, 0.5, "Lower cid, but rank 2"))  # bad
    for i in range(3, 101):
        rows.append((f"CAND_{i:07d}", i, 0.5 - (i - 2) * 0.001, "X"))
    output.write_submission(rows, path)
    errors = output.validate(path)
    assert any("tie-break" in e.lower() for e in errors), f"expected tie-break error: {errors}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        import sys
        sys.exit(1)
