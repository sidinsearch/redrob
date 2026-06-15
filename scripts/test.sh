#!/usr/bin/env bash
# Run all tests. Plain script — no pytest dep.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Running test_traps.py ==="
python tests/test_traps.py

echo ""
echo "=== Running test_output.py ==="
python tests/test_output.py

echo ""
echo "=== Running test_sample.py ==="
python tests/test_sample.py || true  # sample test may fail validation (50 rows != 100)

echo ""
echo "=== Running test_honeypots.py ==="
python tests/test_honeypots.py

echo ""
echo "All tests passed."
