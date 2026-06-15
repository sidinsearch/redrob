# Redrob Candidate Ranker — reproducible container
#
# Build:
#   docker build -t redrob-ranker .
#
# Run ranker (CPU only, no network):
#   docker run --rm \
#     -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
#     -v /path/to/output:/output \
#     redrob-ranker \
#     python rank.py --candidates /data/candidates.jsonl --out /output/submission.csv
#
# Run sandbox (Streamlit):
#   docker run --rm -p 8501:8501 redrob-ranker streamlit run app.py
#
# Constraints satisfied:
# - CPU only (no --gpus)
# - No network during ranking
# - <1 GB image (alpine base + stdlib + streamlit)
# - Runtime <1 min on a 16 GB machine

FROM python:3.11-alpine

# Don't buffer stdout/stderr (so logs show up in real time)
ENV PYTHONUNBUFFERED=1

# Streamlit config (silent)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# App dir
WORKDIR /app

# Install only streamlit (ranker is stdlib-only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY rank.py .
COPY app.py .
COPY src/ ./src/
COPY tests/ ./tests/

# Create output dir
RUN mkdir -p /output

# Default to running the ranker (override with streamlit run app.py for the sandbox)
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/output/submission.csv"]
