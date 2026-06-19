"""app.py — Professional Streamlit sandbox for the Redrob candidate ranker.

Interactive ranking interface with:
- Process ALL uploaded candidates (no 100-limit)
- Interactive weight tuning with real-time re-ranking
- Score breakdown visualizations
- Trap detection with evidence
- Candidate comparison mode
- Fairness audit (distribution analysis)
- Professional UI with custom styling

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Make src/ importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from features import extract_features
from scoring import compute_score
from trap_detector import analyze
from reasoning import generate_reasoning


# ----------------------------------------------------------------------------
# Page config & custom CSS
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional custom styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .stApp header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        padding: 15px;
        font-weight: 600;
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .custom-header p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    /* Trap badges */
    .trap-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 2px;
    }
    
    .trap-honeypot { background: #fee2e2; color: #991b1b; }
    .trap-stuffer { background: #fef3c7; color: #92400e; }
    .trap-template { background: #dbeafe; color: #1e40af; }
    .trap-consulting { background: #e0e7ff; color: #3730a3; }
    .trap-chaser { background: #fce7f3; color: #9d174d; }
    .trap-clean { background: #d1fae5; color: #065f46; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading (no hardcoded paths)
# ----------------------------------------------------------------------------

@st.cache_data
def load_sample_data() -> List[dict]:
    """Load bundled sample candidates from relative path."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "sample_candidates.jsonl",
        Path(__file__).parent.parent.parent / "sample_candidates.jsonl",
        Path("sample_candidates.jsonl"),
    ]
    
    for sample_path in possible_paths:
        if sample_path.exists():
            text = sample_path.read_text(encoding="utf-8").strip()
            if text.startswith("["):
                return json.loads(text)
            else:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
    
    # Return empty list if no sample found
    return []


def parse_uploaded_file(uploaded_file) -> List[dict]:
    """Parse uploaded JSONL or JSON array file. Cached so re-renders don't re-parse."""
    return _parse_uploaded_file_cached(uploaded_file.id, uploaded_file.getvalue())


@st.cache_data(show_spinner=False)
def _parse_uploaded_file_cached(_file_id: str, content: bytes) -> List[dict]:
    """Internal cached parser. The file_id is a cache-key hint; content is the bytes."""
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    text = content.strip()
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    else:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ----------------------------------------------------------------------------
# Ranking pipeline
# ----------------------------------------------------------------------------

def rank_with_weights(
    candidates: List[dict],
    weights: Dict[str, float],
    top_k: int = 100
) -> tuple[List[dict], Dict[str, Any], List[dict]]:
    """Rank candidates with custom weights. Returns (rows, stats, honeypots).

    honeypots: list of dicts with full details of every detected honeypot.
    """
    # Update config weights temporarily
    original_weights = config.WEIGHTS.copy()
    config.WEIGHTS.update(weights)

    try:
        scored = []
        trap_stats = {
            "total_honeypots": 0,
            "total_keyword_stuffers": 0,
            "total_template_summary": 0,
            "total_consulting_only": 0,
            "total_title_chaser": 0,
            "total_clean": 0,
        }
        honeypot_rows: List[dict] = []

        for cand in candidates:
            f = extract_features(cand)
            t = analyze(f)

            # Count traps
            if t.is_honeypot:
                trap_stats["total_honeypots"] += 1
                # Capture full honeypot details
                honeypot_rows.append({
                    "candidate_id": f.candidate_id,
                    "current_title": f.current_title,
                    "current_company": f.current_company,
                    "years_of_experience": f.years_of_experience,
                    "location": f.location,
                    "country": f.country,
                    "honeypot_reasons": " | ".join(t.honeypot_reasons),
                    "n_reasons": len(t.honeypot_reasons),
                    "career_timeline_anomaly": int(f.career_timeline_anomaly),
                    "expert_with_zero_duration": int(f.expert_with_zero_duration),
                    "title_skills_history_mismatch": int(f.title_skills_history_mismatch),
                    "employment_overlap_anomaly": int(f.employment_overlap_anomaly),
                    "duration_integrity_violation": int(f.duration_integrity_violation),
                    "title_responsibility_mismatch": int(f.title_responsibility_mismatch),
                    "skill_experience_contradiction": int(f.skill_experience_contradiction),
                    "education_timeline_anomaly": int(f.education_timeline_anomaly),
                    "career_progression_anomaly": int(f.career_progression_anomaly),
                    "achievement_inflation": int(f.achievement_inflation),
                    "technology_age_anomaly": int(f.technology_age_anomaly),
                    "synthetic_profile": int(f.synthetic_profile),
                    "cross_field_inconsistency": int(f.cross_field_inconsistency),
                    "nlp_claim_without_evidence": int(f.nlp_claim_without_evidence),
                    "ai_skill_count": f.ai_skill_count,
                    "pre_llm_roles": f.pre_llm_roles,
                    "num_career_entries": len(f.career_history),
                })
            if t.is_keyword_stuffer:
                trap_stats["total_keyword_stuffers"] += 1
            if t.is_template_summary:
                trap_stats["total_template_summary"] += 1
            if t.is_consulting_only:
                trap_stats["total_consulting_only"] += 1
            if t.is_title_chaser:
                trap_stats["total_title_chaser"] += 1
            if not any([t.is_honeypot, t.is_keyword_stuffer, t.is_template_summary,
                       t.is_consulting_only, t.is_title_chaser]):
                trap_stats["total_clean"] += 1

            score = compute_score(f, t)
            scored.append((score, cand.get("candidate_id", ""), cand, f, t))

        # Sort and take top-K
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:top_k]

        # Build output rows
        rows = []
        for rank, (score, cid, cand, f, t) in enumerate(top, start=1):
            reasoning = generate_reasoning(f, t, rank)
            rows.append({
                "candidate_id": cid,
                "rank": rank,
                "score": score,
                "reasoning": reasoning,
                "features": f,
                "traps": t,
                "candidate": cand,
            })

        trap_stats["in_topk"] = sum(1 for r in rows if r["traps"].is_honeypot)

        return rows, trap_stats, honeypot_rows

    finally:
        # Restore original weights
        config.WEIGHTS.clear()
        config.WEIGHTS.update(original_weights)



def main():
    # ----------------------------------------------------------------------------
    # Main UI
    # ----------------------------------------------------------------------------

    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1>🎯 Redrob Candidate Ranker</h1>
        <p>Intelligent candidate discovery & ranking for the Senior AI Engineer — Founding Team JD</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
    
        # Data source
        data_source = st.radio(
            "Data source",
            options=["Upload JSONL file", "Use sample data (if available)"],
            index=0,
            help="Upload your own candidates.jsonl or use the bundled sample"
        )
    
        st.divider()
    
        # Top-K selector
        top_k = st.slider(
            "Top-K candidates",
            min_value=10,
            max_value=100,
            value=100,
            step=10,
            help="Number of top candidates to select"
        )
    
        st.divider()
    
        # Interactive weight tuning
        st.markdown("### 🎚️ Weight Tuning")
        st.markdown("Adjust scoring weights (must sum to 1.0)")

        weights = {}
        weight_names = [
            ("career_history_relevance", "Career history relevance", 0.40),
            ("project_impact", "Project impact (evidence)", 0.20),
            ("skills", "Skills (supporting signal)", 0.15),
            ("availability", "Availability", 0.10),
            ("company_quality", "Company quality", 0.10),
            ("education", "Education", 0.05),
        ]

        for key, label, default in weight_names:
            weights[key] = st.slider(
                label,
                min_value=0.0,
                max_value=0.5,
                value=default,
                step=0.01,
                key=f"weight_{key}"
            )
    
        # Normalize weights
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.2f} (should be 1.0). Auto-normalizing...")
            weights = {k: v / total for k, v in weights.items()}
    
        st.divider()
    
        # Pipeline info
        with st.expander("📊 Pipeline Details", expanded=False):
            st.markdown("""
            **6-stage pipeline:**
            1. **Parse** — JSONL streaming
            2. **Features** — 40+ signals per candidate
            3. **Traps** — 4 trap types + 3 honeypot patterns
            4. **Score** — weighted composite × trap multiplier
            5. **Rank** — top-K by score desc, cid asc
            6. **Reason** — 8 templates, 4 rank tiers
        
            **Constraints:**
            - ✅ Stdlib-only (no external deps)
            - ✅ CPU-only (no GPU)
            - ✅ No network calls
            - ✅ <5 min runtime
            """)
    
        with st.expander("🛡️ Trap Legend", expanded=False):
            st.markdown("""
            <div class="trap-badge trap-honeypot">🍯 Honeypot</div> Impossible profile<br>
            <div class="trap-badge trap-stuffer">🔤 Keyword stuffer</div> Non-tech + AI buzzwords<br>
            <div class="trap-badge trap-template">📋 Template summary</div> Canned "curious about AI"<br>
            <div class="trap-badge trap-consulting">🏢 Consulting only</div> TCS/Infosys/Wipro career<br>
            <div class="trap-badge trap-chaser">🔀 Title chaser</div> 4+ jobs in 8yr, avg <18mo<br>
            <div class="trap-badge trap-clean">✓ Clean</div> No traps detected<br>
            """, unsafe_allow_html=True)


    # ----------------------------------------------------------------------------
    # Data ingestion
    # ----------------------------------------------------------------------------

    candidates = []

    if data_source == "Upload JSONL file":
        uploaded = st.file_uploader(
            "Upload candidates JSONL",
            type=["jsonl", "json"],
            help="Upload candidates.jsonl (any size — we'll process all and output top-100)"
        )

        if uploaded is None:
            st.info("📤 Upload a JSONL file to begin ranking")
            st.stop()

        # Cached parser: same file → same candidates, no re-parse on tab change
        with st.spinner("Parsing uploaded file..."):
            candidates = parse_uploaded_file(uploaded)

        st.success(f"✅ Loaded **{len(candidates):,}** candidates")

    else:
        with st.spinner("Loading sample data..."):
            candidates = load_sample_data()

        if not candidates:
            st.error("❌ No sample data found. Please upload a JSONL file.")
            st.stop()

        st.success(f"✅ Loaded **{len(candidates)}** sample candidates")


    # ----------------------------------------------------------------------------
    # Run ranking (cached: only re-rank when weights / top_k / source change)
    # ----------------------------------------------------------------------------

    st.divider()

    # Build a stable cache key from the data source + weights + top_k.
    # session_state is keyed by the upload file id (or 'sample' for sample data).
    if data_source == "Upload JSONL file" and uploaded is not None:
        cache_source_key = uploaded.file_id if hasattr(uploaded, "file_id") else "uploaded"
    else:
        cache_source_key = "sample"
    cache_key = (cache_source_key, tuple(sorted(weights.items())), top_k)

    # Only run ranking if cache_key changed.
    if (
        "rank_cache_key" not in st.session_state
        or st.session_state.rank_cache_key != cache_key
        or "rank_rows" not in st.session_state
    ):
        with st.spinner(f"Ranking {len(candidates):,} candidates..."):
            t0 = time.perf_counter()
            rows, trap_stats, honeypot_rows = rank_with_weights(candidates, weights, top_k=top_k)
            elapsed = time.perf_counter() - t0
        st.session_state.rank_cache_key = cache_key
        st.session_state.rank_rows = rows
        st.session_state.rank_trap_stats = trap_stats
        st.session_state.rank_honeypot_rows = honeypot_rows
        st.session_state.rank_candidates = candidates
        st.session_state.rank_elapsed = elapsed
    else:
        rows = st.session_state.rank_rows
        trap_stats = st.session_state.rank_trap_stats
        honeypot_rows = st.session_state.rank_honeypot_rows
        candidates = st.session_state.rank_candidates
        elapsed = st.session_state.rank_elapsed

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Candidates", f"{len(candidates):,}")
    col2.metric("Top-K", len(rows))
    col3.metric("Runtime", f"{elapsed:.2f}s")
    col4.metric("Honeypots in Top-100", trap_stats.get("in_topk", 0))
    col5.metric("Total Honeypots (excluded)", len(honeypot_rows))


    # ----------------------------------------------------------------------------
    # Tabs for different views
    # ----------------------------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Top Candidates",
        "🛡️ Trap Analysis",
        "📊 Score Breakdown",
        "🔍 Candidate Drill-down",
        "⚖️ Fairness Audit",
        "🍯 Excluded Honeypots"
    ])


    # ----------------------------------------------------------------------------
    # Tab 1: Top Candidates
    # ----------------------------------------------------------------------------

    with tab1:
        st.markdown("### 🏆 Top Ranked Candidates")
    
        # Create DataFrame for display
        df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "ID": r["candidate_id"],
                "Score": f"{r['score']:.4f}",
                "Reasoning": r["reasoning"],
            }
            for r in rows
        ])
    
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=500,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "ID": st.column_config.TextColumn("Candidate ID", width="medium"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            }
        )
    
        # Download buttons
        col1, col2 = st.columns(2)
    
        with col1:
            # CSV download
            csv_buf = io.StringIO()
            import csv as _csv
            writer = _csv.writer(csv_buf, lineterminator="\n")
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for r in rows:
                reasoning = (r["reasoning"] or "").replace("\n", " ").replace("\r", " ").strip()
                if len(reasoning) > 300:
                    reasoning = reasoning[:299].rstrip() + "…"
                writer.writerow([r["candidate_id"], int(r["rank"]), f"{float(r['score']):.6f}", reasoning])
        
            st.download_button(
                label="📥 Download CSV",
                data=csv_buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
        with col2:
            # JSON download
            json_data = json.dumps([
                {
                    "candidate_id": r["candidate_id"],
                    "rank": r["rank"],
                    "score": r["score"],
                    "reasoning": r["reasoning"],
                }
                for r in rows
            ], indent=2)
        
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="ranking.json",
                mime="application/json",
                use_container_width=True,
            )


    # ----------------------------------------------------------------------------
    # Tab 2: Trap Analysis
    # ----------------------------------------------------------------------------

    with tab2:
        st.markdown("### 🛡️ Trap Detection Analysis")
    
        # Trap distribution chart
        trap_data = {
            "Honeypots": trap_stats.get("total_honeypots", 0),
            "Keyword Stuffers": trap_stats.get("total_keyword_stuffers", 0),
            "Template Summary": trap_stats.get("total_template_summary", 0),
            "Consulting Only": trap_stats.get("total_consulting_only", 0),
            "Title Chasers": trap_stats.get("total_title_chaser", 0),
            "Clean": trap_stats.get("total_clean", 0),
        }
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### Trap Distribution")
            fig = px.pie(
                values=list(trap_data.values()),
                names=list(trap_data.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
        with col2:
            st.markdown("#### Trap Counts")
            fig = go.Figure(data=[
                go.Bar(
                    x=list(trap_data.keys()),
                    y=list(trap_data.values()),
                    marker_color=['#ef4444', '#f59e0b', '#3b82f6', '#6366f1', '#ec4899', '#10b981']
                )
            ])
            fig.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
        # Detailed trap table
        st.markdown("#### Candidates with Traps")
    
        trap_rows = [r for r in rows if any([
            r["traps"].is_honeypot,
            r["traps"].is_keyword_stuffer,
            r["traps"].is_template_summary,
            r["traps"].is_consulting_only,
            r["traps"].is_title_chaser,
        ])]
    
        if trap_rows:
            trap_df = pd.DataFrame([
                {
                    "Rank": r["rank"],
                    "ID": r["candidate_id"],
                    "Score": f"{r['score']:.4f}",
                    "Trap Multiplier": f"{r['traps'].trap_multiplier:.3f}",
                    "Flags": ", ".join([
                        f for f, v in [
                            ("🍯 Honeypot", r["traps"].is_honeypot),
                            ("🔤 Stuffer", r["traps"].is_keyword_stuffer),
                            ("📋 Template", r["traps"].is_template_summary),
                            ("🏢 Consulting", r["traps"].is_consulting_only),
                            ("🔀 Chaser", r["traps"].is_title_chaser),
                        ] if v
                    ]) or "✓ Clean",
                }
                for r in trap_rows
            ])
            st.dataframe(trap_df, width="stretch", hide_index=True)
        else:
            st.success("✅ No traps detected in top candidates!")


    # ----------------------------------------------------------------------------
    # Tab 3: Score Breakdown
    # ----------------------------------------------------------------------------

    with tab3:
        st.markdown("### 📊 Score Component Breakdown (Evidence-based)")

        # Select candidate for breakdown
        selected_id = st.selectbox(
            "Select candidate",
            [r["candidate_id"] for r in rows],
            format_func=lambda x: f"Rank {next(r['rank'] for r in rows if r['candidate_id'] == x)}: {x}"
        )

        if selected_id:
            row = next(r for r in rows if r["candidate_id"] == selected_id)
            f = row["features"]

            # Compute the new evidence-based component scores (matching scoring.py)
            from scoring import (
                _career_history_relevance, _project_impact, _skills_score,
                _availability_score, _company_quality_score, compute_relevance,
                compute_authenticity,
            )
            from trap_detector import TrapInfo
            trap = row["traps"]
            # Quick trap info stub for company_quality
            trap_stub = type("T", (), {
                "is_consulting_only": trap.is_consulting_only,
                "is_keyword_stuffer": False,
                "is_template_summary": False,
            })()

            chr_score = _career_history_relevance(f)
            pi_score = _project_impact(f)
            sk_score = _skills_score(f)
            av_score = _availability_score(f)
            cq_score = _company_quality_score(f, trap_stub)
            ed_score = f.education_score

            w = weights
            components = {
                "Career history relevance (40%)": chr_score * w.get("career_history_relevance", 0.40),
                "Project impact (20%)": pi_score * w.get("project_impact", 0.20),
                "Skills (15%)": sk_score * w.get("skills", 0.15),
                "Availability (10%)": av_score * w.get("availability", 0.10),
                "Company quality (10%)": cq_score * w.get("company_quality", 0.10),
                "Education (5%)": ed_score * w.get("education", 0.05),
            }

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Component Contributions")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(components.keys()),
                        y=list(components.values()),
                        marker_color=px.colors.qualitative.Pastel,
                    )
                ])
                fig.update_layout(
                    xaxis_tickangle=-45,
                    yaxis_title="Weighted Score Contribution",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Score Distribution")
                fig = px.pie(
                    values=list(components.values()),
                    names=list(components.keys()),
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.3,
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Evidence flags (Rule 6: high-value signals)
            st.markdown("#### Project Impact Evidence (Rule 6)")
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            with ev_col1:
                st.markdown(f"**Ranking evidence:** {'✅' if f.has_ranking_evidence else '❌'}")
                st.markdown(f"**Retrieval evidence:** {'✅' if f.has_retrieval_evidence else '❌'}")
            with ev_col2:
                st.markdown(f"**Eval evidence:** {'✅' if f.has_eval_evidence else '❌'}")
                st.markdown(f"**A/B test evidence:** {'✅' if f.has_ab_test_evidence else '❌'}")
            with ev_col3:
                st.markdown(f"**Production evidence:** {'✅' if f.has_production_evidence else '❌'}")
                st.markdown(f"**High-value roles:** {f.high_value_role_count}")

            # Raw scores table
            st.markdown("#### Raw Component Scores")
            raw_df = pd.DataFrame([
                {"Component": k, "Raw Score (0-1)": f"{v:.3f}", "Weight": f"{w.get(key, 0):.2f}"}
                for k, v, key in [
                    ("Career history relevance", chr_score, "career_history_relevance"),
                    ("Project impact (evidence)", pi_score, "project_impact"),
                    ("Skills", sk_score, "skills"),
                    ("Availability", av_score, "availability"),
                    ("Company quality", cq_score, "company_quality"),
                    ("Education", ed_score, "education"),
                ]
            ])
            st.dataframe(raw_df, width="stretch", hide_index=True)


    # ----------------------------------------------------------------------------
    # Tab 4: Candidate Drill-down
    # ----------------------------------------------------------------------------

    with tab4:
        st.markdown("### 🔍 Candidate Deep Dive")
    
        selected_id = st.selectbox(
            "Select candidate",
            [r["candidate_id"] for r in rows],
            format_func=lambda x: f"Rank {next(r['rank'] for r in rows if r['candidate_id'] == x)}: {x}",
            key="drilldown_select"
        )
    
        if selected_id:
            row = next(r for r in rows if r["candidate_id"] == selected_id)
            f = row["features"]
            t = row["traps"]
            cand = row["candidate"]
        
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown("#### 📋 Profile")
                st.markdown(f"""
                - **Title:** {f.current_title}
                - **Company:** {f.current_company} ({f.current_industry})
                - **YoE:** {f.years_of_experience:.1f} years
                - **Location:** {f.location}, {f.country}
                - **Notice period:** {f.notice_period_days} days
                - **Open to work:** {"✅ Yes" if f.open_to_work else "❌ No"}
                - **Verified:** {"✅" if f.verified_email else "❌"} email, {"✅" if f.verified_phone else "❌"} phone
                """)
        
            with col2:
                st.markdown("#### 🛡️ Trap Status")
            
                trap_labels = []
                if t.is_honeypot:
                    trap_labels.append('<span class="trap-badge trap-honeypot">🍯 HONEYPOT</span>')
                if t.is_keyword_stuffer:
                    trap_labels.append('<span class="trap-badge trap-stuffer">🔤 Keyword stuffer</span>')
                if t.is_template_summary:
                    trap_labels.append('<span class="trap-badge trap-template">📋 Template summary</span>')
                if t.is_consulting_only:
                    trap_labels.append('<span class="trap-badge trap-consulting">🏢 Consulting only</span>')
                if t.is_title_chaser:
                    trap_labels.append('<span class="trap-badge trap-chaser">🔀 Title chaser</span>')
                if not trap_labels:
                    trap_labels.append('<span class="trap-badge trap-clean">✓ Clean</span>')
            
                st.markdown(" ".join(trap_labels), unsafe_allow_html=True)
                st.markdown(f"**Trap multiplier:** `{t.trap_multiplier:.3f}`")
            
                if t.honeypot_reasons:
                    st.markdown("**Reasons:**")
                    for reason in t.honeypot_reasons:
                        st.markdown(f"- {reason}")
        
            # Skills
            st.markdown("#### 🎯 Skills")
            skills = cand.get("skills", [])
            if skills:
                skill_df = pd.DataFrame([
                    {
                        "Skill": s.get("name", ""),
                        "Proficiency": s.get("proficiency", ""),
                        "Endorsements": s.get("endorsements", 0),
                        "Duration (months)": s.get("duration_months", 0),
                    }
                    for s in skills[:20]  # Show top 20
                ])
                st.dataframe(skill_df, width="stretch", hide_index=True)
            else:
                st.info("No skills listed")
        
            # Career history
            st.markdown("#### 💼 Career History")
            career = cand.get("career_history", [])
            if career:
                career_df = pd.DataFrame([
                    {
                        "Company": c.get("company", ""),
                        "Title": c.get("title", ""),
                        "Duration": f"{c.get('duration_months', 0)} months",
                        "Industry": c.get("industry", ""),
                    }
                    for c in career
                ])
                st.dataframe(career_df, width="stretch", hide_index=True)
        
            # Reasoning
            st.markdown("#### 💭 Generated Reasoning")
            st.info(row["reasoning"])


    # ----------------------------------------------------------------------------
    # Tab 5: Fairness Audit
    # ----------------------------------------------------------------------------

    with tab5:
        st.markdown("### ⚖️ Fairness & Diversity Audit")
    
        # Extract features for all top candidates
        top_candidates = [r["candidate"] for r in rows]
    
        # Country distribution
        countries = [c.get("profile", {}).get("country", "Unknown") for c in top_candidates]
        country_counts = pd.Series(countries).value_counts()
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### 🌍 Geographic Distribution")
            fig = px.pie(
                values=country_counts.values,
                names=country_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
        # Education tier distribution
        tiers = []
        for c in top_candidates:
            edu = c.get("education", [])
            if edu:
                tier = edu[0].get("tier", "unknown")
                tiers.append(tier)
            else:
                tiers.append("unknown")
    
        tier_counts = pd.Series(tiers).value_counts()
    
        with col2:
            st.markdown("#### 🎓 Education Tier Distribution")
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
        # YoE distribution
        yoe_values = [c.get("profile", {}).get("years_of_experience", 0) for c in top_candidates]
    
        st.markdown("#### 📅 Years of Experience Distribution")
        fig = px.histogram(
            x=yoe_values,
            nbins=20,
            labels={"x": "Years of Experience", "y": "Count"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
        # Summary stats
        st.markdown("#### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg YoE", f"{sum(yoe_values) / len(yoe_values):.1f}")
        col2.metric("Countries", len(country_counts))
        col3.metric("Education Tiers", len(tier_counts))
        col4.metric("Avg Score", f"{sum(r['score'] for r in rows) / len(rows):.3f}")


    # ----------------------------------------------------------------------------
    # Tab 6: Excluded Honeypots (full pool, NOT just top-100)
    # ----------------------------------------------------------------------------
    with tab6:
        st.markdown("### 🍯 Excluded Honeypots (Full Pool)")
        st.caption(
            "Candidates with impossibly wrong profiles that were excluded from ranking. "
            "These are detected across the entire candidate pool (not just the top-100). "
            "All are forced to a score of -1e9 and will never appear in the ranked output."
        )
        n_hp = len(honeypot_rows)
        if n_hp == 0:
            st.success("🎉 No honeypots detected in the candidate pool!")
        else:
            # Top-line metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Honeypots", f"{n_hp:,}")
            col2.metric("Honeypot Rate", f"{n_hp / max(1, len(candidates)) * 100:.2f}%")
            col3.metric("In Top-100", "0" if trap_stats.get("in_topk", 0) == 0 else f"{trap_stats['in_topk']}")
            col4.metric("Detectors Active", f"{sum(1 for k in ['career_timeline_anomaly','expert_with_zero_duration','title_skills_history_mismatch','employment_overlap_anomaly','duration_integrity_violation','title_responsibility_mismatch','skill_experience_contradiction','education_timeline_anomaly','career_progression_anomaly','achievement_inflation','technology_age_anomaly','synthetic_profile','cross_field_inconsistency','nlp_claim_without_evidence'] if any(r.get(k, 0) for r in honeypot_rows))}/14")

            # Detector breakdown
            st.markdown("#### Detection breakdown")
            detector_cols = {
                "Career Timeline": "career_timeline_anomaly",
                "Expert w/ 0 Duration": "expert_with_zero_duration",
                "Title-Skills Mismatch": "title_skills_history_mismatch",
                "Employment Overlap": "employment_overlap_anomaly",
                "Duration Integrity": "duration_integrity_violation",
                "Title-Responsibility": "title_responsibility_mismatch",
                "Skill-Experience": "skill_experience_contradiction",
                "Education Timeline": "education_timeline_anomaly",
                "Career Progression": "career_progression_anomaly",
                "Achievement Inflation": "achievement_inflation",
                "Tech Age Anomaly": "technology_age_anomaly",
                "Synthetic Profile": "synthetic_profile",
                "Cross-Field Inconsistent": "cross_field_inconsistency",
                "NLP Self-Contradiction": "nlp_claim_without_evidence",
            }
            counts = {label: sum(1 for r in honeypot_rows if r.get(col, 0)) for label, col in detector_cols.items()}
            # Render as a small bar chart
            count_df = pd.DataFrame({"Detector": list(counts.keys()), "Count": list(counts.values())}).sort_values("Count", ascending=False)
            count_df = count_df[count_df["Count"] > 0]
            if not count_df.empty:
                fig = px.bar(count_df, x="Count", y="Detector", orientation="h", color="Count", color_continuous_scale="Reds")
                fig.update_layout(height=350, showlegend=False, yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No individual detector fires — check the candidate set.")

            # Full honeypot table
            st.markdown(f"#### Full honeypot details ({n_hp:,} candidates)")
            display_df = pd.DataFrame(honeypot_rows)
            # Show key columns by default
            show_cols = ["candidate_id", "current_title", "current_company", "years_of_experience",
                         "country", "n_reasons", "honeypot_reasons"]
            available = [c for c in show_cols if c in display_df.columns]
            st.dataframe(display_df[available], use_container_width=True, height=400)

            # Download honeypots.csv
            csv_buf = io.StringIO()
            display_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download honeypots.csv",
                data=csv_buf.getvalue(),
                file_name="honeypots.csv",
                mime="text/csv",
            )

            # Drill-down on a single honeypot
            st.markdown("#### Honeypot drill-down")
            selected_hp = st.selectbox(
                "Inspect a honeypot",
                options=[r["candidate_id"] for r in honeypot_rows[:200]] if len(honeypot_rows) > 200 else [r["candidate_id"] for r in honeypot_rows],
                key="honeypot_select",
            )
            if selected_hp:
                hp_record = next((r for r in honeypot_rows if r["candidate_id"] == selected_hp), None)
                if hp_record:
                    st.markdown(f"**{hp_record['current_title']}** at **{hp_record['current_company']}** — {hp_record['years_of_experience']} YoE ({hp_record.get('location', '')}, {hp_record.get('country', '')})")
                    reasons = hp_record.get("honeypot_reasons", "")
                    if reasons:
                        st.error(f"**Why flagged:** {reasons}")
                    # Show the 14 flags as a small badge grid
                    active_flags = [label for label, col in detector_cols.items() if hp_record.get(col, 0)]
                    if active_flags:
                        st.markdown("**Active detectors:**")
                        st.write(", ".join(f"`{f}`" for f in active_flags))
                    # Find the full candidate record to show career/skills
                    full_cand = next((c for c in candidates if c.get("candidate_id") == selected_hp), None)
                    if full_cand:
                        with st.expander("Full profile"):
                            st.json({
                                "current_title": full_cand.get("profile", {}).get("current_title"),
                                "current_company": full_cand.get("profile", {}).get("current_company"),
                                "years_of_experience": full_cand.get("profile", {}).get("years_of_experience"),
                                "summary": (full_cand.get("profile", {}).get("summary") or "")[:500],
                                "num_skills": len(full_cand.get("skills", [])),
                                "num_career_entries": len(full_cand.get("career_history", [])),
                            })


    # ----------------------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------------------

    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p><strong>Redrob Candidate Ranker</strong> — Built for the Redrob AI Challenge</p>
        <p>Stdlib-only • CPU-only • No network • <5 min runtime • 0% honeypot rate</p>
        <p>Code: <code>rank.py</code> • <code>src/</code> • <code>docs/</code></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
