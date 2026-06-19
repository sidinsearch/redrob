# Redrob Candidate Ranker — reproducible container.
#
# Build once, then pick a mode on `docker run`:
#
#   # Streamlit sandbox (default — browser opens on http://localhost:8501)
#   docker run --rm -p 8501:8501 redrob-ranker
#
#   # Or with your data pre-loaded
#   docker run --rm -p 8501:8501 \
#     -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
#     redrob-ranker
#
#   # CLI ranking (headless, one-shot)
#   docker run --rm \
#     -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
#     -v /path/to/output:/output \
#     redrob-ranker \
#     python rank.py --candidates /data/candidates.jsonl \
#                          --out /output/submission.csv

FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install sandbox deps first (best layer caching — changes rarely).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rank.py app.py ./
COPY sample_candidates.jsonl ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY .streamlit/ ./.streamlit/
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh

# /data and /output are the default mount points documented at the top.
RUN mkdir -p /data /output && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8501

# No CMD → entrypoint gets zero args → falls into the Streamlit branch.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
