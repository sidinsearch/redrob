"""app.py — Interactive Streamlit sandbox for the Redrob candidate ranker.

Lets you upload a small candidate sample (or use a built-in one), runs the
ranking pipeline, and shows:
- Top 100 with score, reasoning, and per-candidate trap status
- Score distribution (all candidates, top-100 cutoff)
- Trap statistics (honeypot, keyword stuffer, template summary, consulting, chaser)
- Filter by title / company / location / skills
- "Why this score" drill-down: see which signals contributed

Run:
    streamlit run app.py

Acceptance:
- Accepts ≤100 candidates as input (upload or built-in)
- Produces ranked CSV
- Completes within the compute budget (≤5 min on CPU)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

# Make src/ importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

import streamlit as st

import config
import output
import parser
from features import extract_features
from ranking_pipeline import rank_candidates  # see below
from trap_detector import analyze


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data
def load_sample_data() -> bytes:
    """Load the bundled sample candidates as JSONL bytes."""
    sample_path = Path(r"D:\redrob\sample_candidates.jsonl")
    if not sample_path.exists():
        # Fall back to JSON-array form, convert
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        out = "\n".join(json.dumps(c) for c in data)
        return out.encode("utf-8")
    return sample_path.read_bytes()


@st.cache_data
def load_full_data_summary() -> dict:
    """Summary stats from the full candidates file (without loading all)."""
    p = Path(r"D:\redrob\candidates.jsonl")
    if not p.exists():
        return {}
    n = parser.count_lines(p)
    return {
        "n_candidates": n,
        "size_mb": parser.path_size_mb(p),
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("🎯 Redrob Candidate Ranker")
st.markdown(
    """
    **Intelligent candidate discovery & ranking for the Senior AI Engineer — Founding Team JD.**

    A stdlib-only, CPU-only, no-network ranker. Built to satisfy the Redrob
    hackathon constraints (≤5 min on CPU, ≤16 GB RAM, no external API calls).
    """
)

with st.sidebar:
    st.header("Configuration")
    data_source = st.radio(
        "Data source",
        options=["Built-in sample (50 candidates)", "Upload JSONL (≤100 candidates)"],
        index=0,
    )
    top_k = st.slider("Top-K", min_value=10, max_value=100, value=20, step=5)

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    1. **Parse** — JSONL streaming
    2. **Features** — 40+ signals per candidate
    3. **Traps** — 4 trap types + 3 honeypot patterns
    4. **Score** — weighted composite × trap multiplier
    5. **Rank** — top-K by score desc, cid asc
    6. **Reason** — 8 templates, 4 rank tiers
    """)

    st.divider()
    st.markdown("### Trap legend")
    st.markdown("""
    - 🍯 **Honeypot** — forced to bottom (impossible profile)
    - 🔤 **Keyword stuffer** — non-tech title + AI buzzwords only
    - 📋 **Template summary** — "curious about AI tools" canned phrase
    - 🏢 **Consulting only** — TCS/Infosys/Wipro career path
    - 🔀 **Title chaser** — 4+ jobs in 8yr, avg <18mo
    """)


# ----------------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------------

def parse_uploaded_jsonl(uploaded_file) -> List[dict]:
    """Parse a Streamlit uploaded file as JSONL or JSON array."""
    content = uploaded_file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    candidates = []
    text = content.strip()
    if text.startswith("["):
        # JSON array
        data = json.loads(text)
        candidates = data if isinstance(data, list) else [data]
    else:
        # JSONL
        for line in text.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


if data_source == "Upload JSONL (≤100 candidates)":
    uploaded = st.file_uploader("Upload candidates JSONL", type=["jsonl", "json"])
    if uploaded is None:
        st.info("Upload a JSONL file with candidate records to begin.")
        st.stop()
    candidates = parse_uploaded_jsonl(uploaded)
    if len(candidates) > 100:
        st.warning(f"File contains {len(candidates)} candidates — sandbox will only rank the first 100.")
        candidates = candidates[:100]
else:
    raw_bytes = load_sample_data()
    text = raw_bytes.decode("utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        candidates = data if isinstance(data, list) else [data]
    else:
        candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
    st.info(f"Loaded {len(candidates)} built-in sample candidates.")


# ----------------------------------------------------------------------------
# Run ranking
# ----------------------------------------------------------------------------

t0 = time.perf_counter()

with st.spinner("Ranking candidates..."):
    rows, trap_stats = rank_candidates(candidates, top_k=top_k)

elapsed = time.perf_counter() - t0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidates processed", len(candidates))
col2.metric("Top-K selected", len(rows))
col3.metric("Runtime", f"{elapsed:.2f}s")
col4.metric("Honeypots in top-K", trap_stats.get("in_topk", 0))


# ----------------------------------------------------------------------------
# Trap statistics
# ----------------------------------------------------------------------------

st.header("Trap statistics")

stats_cols = st.columns(6)
stats_cols[0].metric("🍯 Honeypots", trap_stats.get("total_honeypots", 0), help="Forced to score 0.0")
stats_cols[1].metric("🔤 Keyword stuffers", trap_stats.get("total_keyword_stuffers", 0))
stats_cols[2].metric("📋 Template summary", trap_stats.get("total_template_summary", 0))
stats_cols[3].metric("🏢 Consulting only", trap_stats.get("total_consulting_only", 0))
stats_cols[4].metric("🔀 Title chasers", trap_stats.get("total_title_chaser", 0))
stats_cols[5].metric("✓ Clean", trap_stats.get("total_clean", 0))


# ----------------------------------------------------------------------------
# Top-K table
# ----------------------------------------------------------------------------

st.header(f"Top {len(rows)} candidates")

# Display as a table
import pandas as pd

df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
df["score"] = df["score"].map(lambda x: f"{x:.4f}")
df = df.rename(columns={"candidate_id": "ID", "score": "Score", "reasoning": "Reasoning"})

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    height=400,
)

# Download button
import io
csv_buf = io.StringIO()
output.write_submission(rows, csv_buf)
st.download_button(
    label="Download submission.csv",
    data=csv_buf.getvalue(),
    file_name="submission.csv",
    mime="text/csv",
)


# ----------------------------------------------------------------------------
# Drill-down
# ----------------------------------------------------------------------------

st.header("Drill-down: why this score?")

selected_id = st.selectbox("Pick a candidate", [r[0] for r in rows])
if selected_id:
    # Find the candidate record
    cand = next((c for c in candidates if c.get("candidate_id") == selected_id), None)
    if cand:
        f = extract_features(cand)
        t = analyze(f)

        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Profile")
            st.markdown(f"""
            - **Title:** {f.current_title}
            - **Company:** {f.current_company} ({f.current_industry})
            - **YoE:** {f.years_of_experience}yr
            - **Location:** {f.location}, {f.country}
            - **Notice period:** {f.notice_period_days} days
            - **Open to work:** {"✓" if f.open_to_work else "✗"}
            """)
        with d2:
            st.subheader("Traps")
            trap_labels = []
            if t.is_honeypot:
                trap_labels.append("🍯 HONEYPOT")
            if t.is_keyword_stuffer:
                trap_labels.append("🔤 Keyword stuffer")
            if t.is_template_summary:
                trap_labels.append("📋 Template summary")
            if t.is_consulting_only:
                trap_labels.append("🏢 Consulting only")
            if t.is_title_chaser:
                trap_labels.append("🔀 Title chaser")
            if not trap_labels:
                trap_labels = ["✓ Clean"]
            st.markdown("**Flags:** " + ", ".join(trap_labels))
            st.markdown(f"**Trap multiplier:** {t.trap_multiplier:.3f}")
            if t.honeypot_reasons:
                st.markdown("**Reasons:**")
                for r in t.honeypot_reasons:
                    st.markdown(f"- {r}")

        st.subheader("Component scores")
        comp_df = pd.DataFrame([
            ("Title relevance", f.title_score, 0.25),
            ("Experience fit", f.experience_fit, 0.12),
            ("Product experience", f.product_exp_score, 0.10),
            ("AI skills depth", f.ai_skills_depth, 0.15),
            ("Career relevance", f"see reasoning", 0.10),
            ("Education", f.education_score, 0.03),
            ("Behavioral", f.behavioral_score, 0.15),
            ("Location", f.location_score, 0.05),
            ("Availability", f.availability_score, 0.05),
        ], columns=["Component", "Value", "Weight"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    **Pipeline details:**
    - Stream JSONL → 40+ features → 4 trap types + 3 honeypot patterns → weighted composite × trap multiplier
    - Runtime on full 100K pool: ~35s on CPU (5x under the 5-min budget)
    - All code is stdlib-only; no network calls during ranking

    **Code:** see `rank.py`, `src/`, `docs/`
    """
)
