#!/usr/bin/env bash
# Quick-start: produce a submission.csv from candidates.jsonl
# Run from the project root.

set -euo pipefail

# Defaults — override via env vars
CANDIDATES="${CANDIDATES:-./candidates.jsonl}"
OUTPUT="${OUTPUT:-./output/submission.csv}"

if [ ! -f "$CANDIDATES" ]; then
    echo "ERROR: candidates file not found: $CANDIDATES"
    echo "Set CANDIDATES env var or pass it as the first arg."
    echo "Usage: $0 [path/to/candidates.jsonl] [path/to/output.csv]"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "[run.sh] ranking $CANDIDATES -> $OUTPUT"
python rank.py --candidates "$CANDIDATES" --out "$OUTPUT"

echo "[run.sh] validating with the official validator"
python "$(dirname "$0")/../validate_submission.py" "$OUTPUT" || {
    echo "WARNING: official validator not found at expected path; running built-in validator"
    python -c "import sys; sys.path.insert(0, 'src'); import output; errors = output.validate('$OUTPUT'); [print(f'  - {e}') for e in errors]; sys.exit(1 if errors else 0)"
}

echo "[run.sh] done"
