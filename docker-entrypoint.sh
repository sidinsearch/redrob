#!/bin/sh
# Redrob Candidate Ranker — container entrypoint.
#
# Two behaviors:
#   1. With a command → run it as-is (terminal use, e.g. `python rank.py ...`).
#   2. With no command → start the Streamlit sandbox on port 8501
#      (the host opens http://localhost:8501 in a browser).

set -eu

if [ "$#" -eq 0 ]; then
    echo "[entrypoint] No command given — starting Streamlit on http://localhost:8501"
    exec streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
fi

exec "$@"
