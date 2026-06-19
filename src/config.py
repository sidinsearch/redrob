"""config.py — All weights, thresholds, title mappings, skill ontologies.

Every magic number in the ranking system lives here. Reviewers should be able
to read this file end-to-end and predict the system's behavior on any candidate.
Tune weights here; do not sprinkle magic numbers across other modules.

ponytail: this file is the single source of truth for tunable parameters.
- All TitleCategory scores and ExperienceFit thresholds use a small set of
  hand-tuned constants. We accept that these are subjective; the alternative
  (data-driven weights) is out of scope for a CPU-only, stdlib-only ranker.
- Skill ontologies are exhaustive lowercase substring checks. We chose
  substring over regex/word-boundary to keep the runtime flat. The trade-off
  is rare false positives (e.g. "rag" inside "storage"); the matching
  trap_detector layer catches most of these via the title requirement.
"""

from __future__ import annotations

# ============================================================================
# Stage weights — composite score formula
# ============================================================================
# All weights sum to 1.0. final_score = trap_multiplier * weighted_sum.
# Tuned against the rubric (NDCG@10 50%, NDCG@50 30%, MAP 15%, P@10 5%)
# and the trap types described in the architecture doc.

# Weights rebalanced per user spec (2026-06-19).
#
# Career evidence still dominates the fit score, but availability is now
# a MULTIPLICATIVE filter (not part of the additive sum). This means a
# perfect-on-paper candidate with poor availability drops dramatically
# rather than just being a little lower in an additive mix.
#
# Final score = fit_score × availability_multiplier × trap_multiplier
#
# Fit score weights (sum to 1.0):
# - career_history_relevance: 0.45 (was 0.40, +0.05)
# - project_impact: 0.25 (was 0.20, +0.05)
# - skills: 0.20 (was 0.15, +0.05)
# - company_quality: 0.05 (was 0.10, -0.05)
# - education: 0.05 (was 0.05, unchanged)
# - availability: 0.00 (was 0.10, moved to multiplicative filter)
WEIGHTS = {
    "career_history_relevance": 0.45,
    "project_impact": 0.25,
    "skills": 0.20,
    "availability": 0.00,  # used as a multiplicative filter, not in the additive sum
    "company_quality": 0.05,
    "education": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# ============================================================================
# Title relevance — 6-level mapping (1.0 best, 0.05 worst)
# ============================================================================
# Lowercase substring matches. Order in the list doesn't matter; we use
# any-of semantics. A title matching multiple categories takes the max.

TITLE_CATEGORIES = [
    # Core AI/ML — the ideal target.
    (1.00, [
        "ai engineer", "ml engineer", "machine learning engineer",
        "nlp engineer", "applied scientist", "applied ml", "applied ai",
        "data scientist", "deep learning engineer", "computer vision engineer",
        "research engineer", "search engineer", "ranking engineer",
        "recommendation engineer", "recommendation systems engineer",
        "recommender systems engineer",
        "retrieval engineer", "relevance engineer", "personalization engineer",
        "llm engineer", "foundation model engineer", "ml research engineer",
        "ai/ml engineer", "ai/ml scientist", "ai scientist",
        "machine learning scientist", "ai research engineer",
    ]),
    # ML-adjacent SWE — engineers with AI in their scope.
    (0.85, [
        "senior software engineer (ml)", "senior software engineer, ml",
        "backend engineer (ai)", "backend engineer, ai",
        "ml platform engineer", "ai platform engineer",
        "ml infrastructure engineer", "ai infrastructure engineer",
        "mlops engineer", "ml systems engineer", "ai systems engineer",
    ]),
    # Software engineer — general SWE without explicit ML title.
    (0.70, [
        "software engineer", "senior software engineer", "staff software engineer",
        "backend engineer", "full stack developer", "full stack engineer",
        "frontend engineer", "platform engineer", "systems engineer",
    ]),
    # Data/Cloud — adjacent but missing the ML layer.
    (0.60, [
        "data engineer", "data analyst", "analytics engineer",
        "cloud engineer", "devops engineer", "site reliability engineer",
        "infrastructure engineer", "database engineer", "etl engineer",
    ]),
    # Tech-adjacent — technical but not in scope.
    (0.40, [
        "qa engineer", "test engineer", "quality engineer",
        "analytics manager", "engineering manager", "tech lead",
    ]),
    # Non-technical — everything else, including trap categories.
    # (Specific traps are caught at trap_detector; here we just score low.)
    (0.05, [
        # Empty / placeholders handled by default
    ]),
]
DEFAULT_TITLE_SCORE = 0.05

# ============================================================================
# Experience fit — Gaussian-like peak at 6-8 years
# ============================================================================
# 5-9 → 1.0 (sweet spot)
# 4-4.9 or 9.1-12 → 0.80
# 3-3.9 or 12.1-14 → 0.50
# <3 or >14 → 0.20

def experience_fit(yoe: float) -> float:
    if 5.0 <= yoe <= 9.0:
        return 1.0
    if 4.0 <= yoe < 5.0 or 9.0 < yoe <= 12.0:
        return 0.80
    if 3.0 <= yoe < 4.0 or 12.0 < yoe <= 14.0:
        return 0.50
    if yoe < 3.0 or yoe > 14.0:
        return 0.20
    return 0.50  # Defensive default for NaN/edge


# ============================================================================
# Product vs Consulting
# ============================================================================
# Consulting/services companies — penalize. Source: JD says "consulting-only
# careers (TCS, Infosys, Wipro, Accenture) flagged as bad fit".
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "persistent", "ltimindtree",
    "hcl", "tech mahindra", "mphasis", "larsen & toubro infotech", "l&t infotech",
    "genpact", "ibm services", "dxc technology", "hexaware", "cyient",
    "zensar", "ltts", "larsen & toubro technology services",
}
# Product companies — boost. (Not exhaustive; absence of consulting flag
# also counts as product for partial boost.)
PRODUCT_COMPANY_HINTS = {
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "uber",
    "airbnb", "stripe", "linkedin", "twitter", "x corp", "spotify",
    "razorpay", "flipkart", "swiggy", "zomato", "ola", "paytm", "phonepe",
    "cred", "meesho", "nykaa", "byjus", "unacademy", "vedantu", "dream11",
    "freshworks", "zoho", "freshdesk", "postman", "browserstack",
    "wysa", "sugarcane", "wonder", "tata 1mg", "urban company", "rapido",
    "ola electric", "zerodha", "groww", "lenskart", "caratlane", "policybazaar",
    "myntra", "ludo king", "sharechat", "mongo db", "mongodb", "snowflake",
    "databricks", "confluent", "elastic", "mongodb", "atlassian", "figma",
    "notion", "slack", "github", "gitlab", "shopify", "salesforce", "oracle",
    "walmart", "target", "costco", "doordash", "instacart", "lyft", "pinterest",
    "dropbox", "box", "twilio", "stripe", "plaid", "coinbase", "robinhood",
}

PRODUCT_EXP_MIXED = 0.6
PRODUCT_EXP_ALL_PRODUCT = 1.0
PRODUCT_EXP_PURE_CONSULTING = 0.0

# ============================================================================
# Skill ontologies
# ============================================================================
# Each list is lowercase substring matches. A skill is "in category" if
# its name contains any string in the list. Casing is normalized.

# Foundational ML — required for genuine AI/ML signal.
# The trap detection uses absence of these to flag keyword stuffers.
FOUNDATIONAL_ML_SKILLS = [
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "machine learning", "deep learning", "neural network", "neural networks",
    "natural language processing", "nlp",
    "transformers", "hugging face transformers", "huggingface transformers",
    "xgboost", "lightgbm", "catboost",
    "reinforcement learning", "computer vision", "speech recognition",
    "object detection", "image classification", "semantic segmentation",
    "named entity recognition", "machine translation", "speech synthesis",
    "tts", "asr",
]

# Production AI tools — bonus, but not foundational.
PRODUCTION_AI_TOOLS = [
    "mlflow", "kubeflow", "weights & biases", "wandb", "dagster", "airflow",
    "dvc", "bentoml", "seldon", "ray", "triton", "tensorrt", "onnx",
    "kubernetes", "k8s", "docker", "terraform", "argo",
    "prometheus", "grafana",
]

# Ranking/search/retrieval — the JD's "nice to have" but matches the role.
RANKING_SKILLS = [
    "elasticsearch", "opensearch", "solr", "lucene",
    "faiss", "annoy", "hnsw",
    "pinecone", "weaviate", "qdrant", "milvus", "chroma", "vespa",
    "vector search", "vector database", "vector db", "embedding search",
    "approximate nearest neighbor", "nearest neighbor",
    "information retrieval", "semantic search", "bm25", "tf-idf", "tfidf",
    "learning to rank", "learning-to-rank", "l2r",
    "colbert", "sbert", "sentence-bert", "sentence bert",
    "retrieval augmented", "retrieval-augmented",
    "sentence-transformers", "sentence transformers",
    "re-ranking", "reranking", "relevance",
    "search relevance", "ranking",
    "hybrid search", "hybrid retrieval",
]

# LLM-only buzzwords — these are exactly what keyword stuffers list. They
# do NOT count as foundational ML on their own.
LLM_BUZZWORD_SKILLS = [
    "llm", "large language model", "large language models",
    "rag", "retrieval-augmented generation", "langchain", "llamaindex",
    "prompt engineering", "prompt engineering",
    "fine-tuning", "fine tuning", "finetuning",
    "lora", "qlora", "peft", "rlhf",
    "embeddings", "embedding",
    "openai", "anthropic", "claude", "gpt", "chatgpt", "gemini", "bard",
    "huggingface", "hugging face",
    "vector database", "vector store",
]

# ============================================================================
# Career description keywords
# ============================================================================
# These catch "Tier 5" plain-language candidates who built ranking/retrieval
# systems without using the exact buzzword. Used by career_relevance.

CAREER_AI_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "neural networks",
    "natural language processing", "nlp", "computer vision", "speech recognition",
    "speech synthesis", "recommender system", "recommendation system",
    "search engine", "search ranking", "search relevance",
    "retrieval system", "retrieval pipeline", "ranking model", "ranking system",
    "embedding model", "vector search", "vector database",
    "text classification", "sentiment analysis", "entity recognition",
    "knowledge graph", "transformer model", "transformer-based",
    "language model", "language modeling", "language modelling",
    "fine-tuning", "fine tuning", "fine-tuned", "finetuning",
    "prompt engineering", "rag", "retrieval augmented",
    "personalization", "personalisation", "candidate matching", "job matching",
    "information retrieval", "semantic search", "hybrid search",
    "model deployment", "model serving", "inference pipeline",
    "training pipeline", "training infrastructure", "ml infrastructure",
    "ml platform", "mlops", "model monitoring", "model drift",
    "evaluation framework", "offline evaluation", "online evaluation",
    "ndcg", "mrr", "map", "auc", "click-through rate",
    "ctr prediction", "click prediction", "ranking algorithm",
    "ab test", "a/b test", "experiment", "online experiment",
]

# ============================================================================
# Trap detection
# ============================================================================

# Template summary trap — exact phrase from the JD.
TEMPLATE_SUMMARY_PHRASE = "Lately I've been curious about how AI tools could augment my work"

# Keyword stuffer thresholds.
KEYWORD_STUFFER_MIN_AI_SKILLS = 4
KEYWORD_STUFFER_MIN_BUZZWORDS = 4

# Title-chaser detection
TITLE_CHASER_MIN_JOBS = 4
TITLE_CHASER_AVG_TENURE_MONTHS = 18
TITLE_CHASER_LOOKBACK_YEARS = 8

# Skill inflation check
SKILL_INFLATION_THRESHOLD = 17  # >17 skills = suspicious
# Spec: "expert proficiency in 10 skills with 0 years used". The data has
# 8 cases at 5+, 13 more at 3-4, none at 6+. Threshold 3 catches the
# borderline cases (e.g., "HR Manager expert in Webpack with 0 months use")
# which are clearly impossible profiles.
EXPERT_SKILL_FAKE_THRESHOLD = 3  # "expert" in 3+ skills with 0 duration = honeypot

# ============================================================================
# Must-have evidence patterns (Step 2 of the audit spec)
# ============================================================================
# These are the JD's four explicit must-haves. Each one is checked against
# career_history.description, NOT the skills list. A skill-list keyword
# with no career_history evidence does NOT satisfy the must-have.
#
# We use a 3-tier system per must-have:
# - primary patterns: strong evidence in a job description
# - context patterns: supporting evidence (e.g., model names used)
# - exclusions: phrases that DISQUALIFY the must-have (e.g., "platform team
#   handled it" — meaning the candidate did NOT do this work personally)
#
# ponytail: the exclusion list is critical. CAND_0046132 was ranked #1 in
# the previous round because the project_impact detector counted
# "collaborative filtering" as recommendation work, but that's NOT
# embeddings-based retrieval. Without exclusions, the model is fooled.
MUST_HAVE_PATTERNS = {
    # ---------------------------------------------------------------
    # Must-have 1: Production embeddings-based retrieval
    # JD: "Production experience with embeddings-based retrieval systems
    # (sentence-transformers, OpenAI embeddings, BGE, E5, or similar)
    # deployed to real users. We don't care which model — we care that
    # you've handled embedding drift, index refresh, retrieval-quality
    # regression in production."
    # ---------------------------------------------------------------
    "embeddings_retrieval": {
        "primary": [
            "deployed retrieval", "production retrieval", "retrieval in production",
            "shipped retrieval", "embedding-based retrieval", "embeddings-based retrieval",
            "vector search production", "semantic search production",
            "sentence-transformers", "sentence transformers",
            "openai embeddings", "bge", "e5", "embedding model",
            "embedding drift", "index refresh", "retrieval-quality regression",
            "retrieval quality", "retrieval system", "dense retrieval",
            "vector database production", "vector db production",
            "ann index", "approximate nearest neighbor",
            "vector embeddings", "embedding-based search", "semantic retrieval",
            "embedding service", "embedding pipeline",
            "embedding index", "ann search", "vector similarity",
            "embedding model", "embedding store", "vector store",
            "production embeddings", "production vector",
        ],
        "context": [
            "retrieval", "embeddings", "vector search", "semantic search",
            "embedding", "vector index", "ann", "approximate nearest",
        ],
        "exclusions": [
            "platform team handled", "platform team manages",
            "outsourced deployment", "managed by infra",
            "not responsible for deployment", "deployment handled by",
            "collaborative filtering", "matrix factorization",
            "two-tower",  # could be a real signal but we want stronger evidence
        ],
    },
    # ---------------------------------------------------------------
    # Must-have 2: Production vector DB / hybrid search
    # JD: "Production experience with vector databases or hybrid search
    # infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
    # Elasticsearch, FAISS, or something similar."
    # ---------------------------------------------------------------
    "vector_db": {
        "primary": [
            "pinecone", "weaviate", "qdrant", "milvus", "vespa", "chroma",
            "faiss", "elasticsearch cluster", "opensearch cluster",
            "vector index", "vector database", "vector db",
            "hybrid search", "hybrid retrieval", "bm25 + dense", "dense + sparse",
            "elasticsearch", "opensearch", "solr", "lucene",
            "vector store", "ann index", "ann search", "approximate nearest",
            "vector similarity", "hnsw", "ivf", "pq",
        ],
        "context": [
            "vector", "embedding index",
        ],
        "exclusions": [
            "platform team handled", "managed by infra",
            "deployment handled by", "not responsible for",
            "only evaluated", "only tested",
        ],
    },
    # ---------------------------------------------------------------
    # Must-have 3: Eval framework design for ranking
    # JD: "Hands-on experience designing evaluation frameworks for ranking
    # systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test
    # interpretation."
    # ---------------------------------------------------------------
    "ranking_eval": {
        "primary": [
            "ranking evaluation", "ranking eval", "search evaluation",
            "ndcg", "mrr", "map@", "offline-online correlation",
            "offline to online", "online evaluation framework",
            "evaluation framework", "evaluation infrastructure",
            "ab test for ranking", "ab test for search", "ab test for recommendation",
            "experiment framework", "experimentation platform",
            "ab test for rec", "a/b testing ranking",
            "offline evaluation", "online evaluation", "offline/online",
            "offline-online", "labeling pipeline", "relevance labeling",
            "human judgments", "click-through data", "click model",
            "ranking metrics", "relevance metrics",
            "evaluation pipeline", "evaluation harness",
        ],
        "context": [
            "ab test", "a/b test", "online experiment", "experiment",
        ],
        "exclusions": [
            "forecasting model", "time series model", "classification model",
            "cv model", "vision model", "speech model", "robotics model",
            "non-ranking",
        ],
    },
    # ---------------------------------------------------------------
    # Must-have 4: Strong Python in a real system
    # JD: "Strong Python. Yes really, we care about code quality."
    # ---------------------------------------------------------------
    "strong_python": {
        "primary": [
            "fastapi", "django", "flask", "pydantic", "celery",
            "pytest", "asyncio", "sqlalchemy", "python service",
            "python microservice", "python backend", "python api",
            "python pipeline", "python production",
            "python codebase", "python library",
            "python framework", "python code", "wrote python",
        ],
        "context": [
            "python", "py", "pytest",
        ],
        "exclusions": [
            "no python", "minimal python", "python only for scripts",
        ],
    },
}

# Nice-to-haves: add up to +10 total to fit_score, never enough alone to
# overcome a low must-have baseline. Detected in career_history too.
NICE_TO_HAVE_PATTERNS = {
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "fine-tuned",
        "instruction tuning", "rlhf", "dpo",
    ],
    "learning_to_rank": [
        "learning to rank", "learning-to-rank", "l2r", "lambdamart", "ranknet",
        "rankboost", "listwise", "pairwise",
    ],
    "hr_tech_marketplace": [
        "hr-tech", "hr tech", "recruiting tech", "marketplace",
        "candidate matching", "job matching", "talent marketplace",
    ],
    "distributed_systems": [
        "distributed system", "kafka", "spark", "ray", "triton",
        "large-scale inference", "distributed inference",
    ],
    "open_source": [
        "open source", "open-source", "github", "maintainer", "contributor",
    ],
}

# Negative patterns: subtract from baseline.
NEGATIVE_PATTERNS = {
    "platform_team_handled": [
        "platform team handled", "platform team manages",
        "outsourced deployment", "deployment handled by",
        "managed by infra", "infra team handled",
    ],
    "title_chasing": [
        "title chase", "title optimization", "rapid promotion",
    ],
    "closed_source_only": [
        "closed source", "closed-source", "proprietary", "internal tool only",
    ],
    "explicit_gap_admission": [
        "still building depth on", "production handled by the platform",
        "looking to step up", "transitioning into", "no production",
        "no deployment experience",
    ],
}

# Hard disqualifiers (Step 1) — fit_score = 0 if any of these fire.
DISQUALIFIER_PATTERNS = {
    "research_only_no_production": [
        # Career in pure research: postdoc, research scientist, lab. No
        # production deployment evidence anywhere.
    ],  # checked structurally, not by keyword
    "consulting_only": [
        # List at config.CONSULTING_COMPANIES — checked structurally
    ],
    "cv_speech_robotics_primary": [
        # Career in CV, speech, robotics with no NLP/IR/retrieval work
    ],  # checked structurally
    "langchain_only_under_12mo": [
        # AI experience < 12mo, mostly LangChain → OpenAI
    ],  # checked structurally
    "no_production_code_18mo": [
        # Senior engineer, no production code in last 18 months
    ],  # checked structurally
}

# Availability multiplier weights (Step 3 of audit spec).
# This is an additive score, clipped to [0.1, 1.0].
AVAILABILITY_WEIGHTS = {
    "open_to_work": 0.40,
    "notice_period": 0.25,
    "recruiter_response": 0.20,
    "recency": 0.15,
}
assert abs(sum(AVAILABILITY_WEIGHTS.values()) - 1.0) < 1e-9, "AVAILABILITY_WEIGHTS must sum to 1.0"

# Fit-score tier baselines (Step 2 of audit spec).
FIT_TIER_BASELINES = {
    4: (80, 100),  # 4/4 must-haves
    3: (55, 75),   # 3/4
    2: (30, 50),   # 2/4
    1: (10, 25),   # 1/4 — explicitly flagged as not meeting core JD requirements
    0: (0, 10),    # 0/4
}

# ============================================================================
# Project impact evidence patterns
# ============================================================================
# Per user spec (2026-06-18): career evidence > skills. Each pattern is a
# "did they actually do it?" signal in any job description.
# Replaces naive "if Pinecone in skills: +10" with "if they deployed
# retrieval: +25". A candidate who built ranking systems outranks a
# candidate who merely knows Pinecone.
PROJECT_IMPACT_PATTERNS = {
    # Tier 1: Highest-value (Rule 6: search/retrieval/ranking/rec/relevance)
    "deployed_retrieval": [
        "deployed retrieval", "deployed a retrieval", "production retrieval",
        "shipped retrieval", "retrieval system to production", "retrieval in production",
        "scaled retrieval", "retrieval at scale", "hybrid search production",
        "vector search production", "semantic search production",
        "embeddings-based retrieval", "serving embeddings",
    ],
    "built_ranking": [
        "built ranking", "built a ranking", "developed ranking", "designed ranking",
        "ranking system", "ranking pipeline", "ranking model", "ranking algorithm",
        "learning to rank", "learning-to-rank", "l2r", "lambdamart", "ranknet",
        "rankboost", "listwise", "pairwise", "rerank", "re-rank", "re-ranking",
        "search ranking", "ranking service", "ranker",
    ],
    "recommendation_or_personalization": [
        "recommendation system", "recommender system", "recommender",
        "personalization", "personalisation", "candidate matching", "job matching",
        "user matching", "content recommendation", "feed ranking",
        "similar jobs", "similar candidates", "people you may know",
        "collaborative filtering", "matrix factorization", "two-tower",
        "candidate generation",
    ],
    # Tier 2: Evaluation expertise (Rule 5: major ranking factor)
    "improved_ndcg_or_eval": [
        "improved ndcg", "ndcg improvement", "lifted ndcg", "boosted ndcg",
        "improved map", "improved mrr", "improved auc", "improved recall",
        "ndcg", "mrr", "map@", "auc", "evaluation framework",
        "offline evaluation", "online evaluation", "offline-online correlation",
        "evaluation pipeline", "evaluation infrastructure",
    ],
    "ran_ab_tests": [
        "a/b test", "ab test", "a-b test", "online experiment",
        "ramp experiment", "holdout experiment", "interleaving",
        "experiment framework", "experimentation platform",
        "ramped rollout", "ab testing", "controlled experiment",
    ],
    # Tier 3: Search infrastructure
    "search_infrastructure": [
        "elasticsearch cluster", "opensearch cluster", "solr cluster",
        "vector index", "faiss index", "pinecone index", "weaviate instance",
        "qdrant deployment", "milvus cluster", "vespa",
        "hybrid search", "bm25 + dense", "dense + sparse",
        "search infrastructure", "search platform", "search service",
    ],
    # Tier 4: Production deployment (Rule 4: production > certifications)
    "production_ml": [
        "deployed ml", "ml in production", "model in production",
        "production ml", "production model", "serving model", "model serving",
        "mlops", "model deployment", "model monitoring", "model drift",
        "training pipeline", "inference pipeline", "feature pipeline",
        "real-time inference", "batch inference", "online inference",
    ],
    # Tier 5: Scale & impact
    "scale_impact": [
        "million users", "users at scale", "production scale",
        "scaled to", "lowered latency", "improved throughput",
        "reduced inference time", "cut latency", "10ms latency", "100ms",
        "billion requests", "million requests", "million candidates",
    ],
}

# Highest-value categories (Rule 6: search/rec/relevance/matching/personalization)
HIGH_VALUE_CATEGORIES = [
    "search", "retrieval", "ranking", "recommendation", "recommender",
    "relevance", "matching", "personalization", "personalisation",
]

# Score weights for project impact (sum to 1.0)
PROJECT_IMPACT_WEIGHTS = {
    "deployed_retrieval": 0.25,
    "built_ranking": 0.25,
    "recommendation_or_personalization": 0.20,
    "improved_ndcg_or_eval": 0.15,
    "ran_ab_tests": 0.10,
    "search_infrastructure": 0.03,
    "production_ml": 0.01,
    "scale_impact": 0.01,
}

# ============================================================================
# Trap multipliers
# ============================================================================
TRAP_MULTIPLIERS = {
    "none": 1.0,
    "keyword_stuffer": 0.40,
    "template_summary": 0.70,
    "consulting_only": 0.75,
    "title_chaser": 0.85,
}
TRAP_FLOOR = 0.30  # Multiple traps floor at 30%

# ============================================================================
# Honeypot detection
# ============================================================================
HONEYPOT_TAX = -1e9  # Force honeypots to bottom
HONEYPOT_YOE_BUFFER_YEARS = 5  # YoE > career span + 5yr → honeypot

# Pre-LLM-era signals. JD: "people who understood retrieval and ranking
# before it became fashionable". Pre-2020 production experience with
# retrieval/ranking/embeddings/ML is a strong positive signal. Detected by
# looking at start_dates in career_history.
PRE_LLM_CUTOFF_YEAR = 2020  # Anything started before this in retrieval/ML = pre-LLM signal
PRE_LLM_BOOST_MAX = 0.30  # capped contribution to pre_llm_signal component

# Timeline validation thresholds
HONEYPOT_OVERLAP_MAX_JOBS = 5  # >5 concurrent jobs (very conservative)
HONEYPOT_GAP_MAX_DAYS = 180   # >6-month unexplained gap = suspicious if claim continuous

# Skill-experience contradiction: claims a skill with 0 months usage.
# Threshold: 5+ skills with advanced/expert proficiency AND <3mo duration
# AND zero endorsements. Stricter than before to avoid FPs.
HONEYPOT_ADV_NO_DURATION_MIN_SKILLS = 5

# Education timeline: claims degree end_year before age 18 (impossible).
HONEYPOT_EDU_AGE_MIN = 18

# Title-responsibility: title claims ML/AI but no ML/AI keywords in any
# job description across career. Only flag if title is strongly AI AND
# ZERO AI keywords anywhere in the entire career. Stricter to avoid FPs.
HONEYPOT_TITLE_RESPONSIBILITY_MATCH_THRESHOLD = 0  # 0 AI keywords = honeypot

# Achievement validation: descriptions with extreme inflation markers
# (e.g., "10x", "100x improvement", "world-class") with no supporting
# metrics. Stricter: 5+ inflation keywords (was 3).
HONEYPOT_ACHIEVEMENT_INFLATION_KEYWORDS = [
    "world-class", "world class", "best in class", "industry-leading",
    "industry leading", "10x faster", "10x improvement", "10x more",
    "revolutionized", "revolutionary", "groundbreaking", "cutting-edge",
    "state-of-the-art", "state of the art", "pioneering", "unprecedented",
]
HONEYPOT_ACHIEVEMENT_INFLATION_MIN = 5  # ≥5 inflation keywords (was 3)

# Synthetic profile: tighter. Spec says ~80 honeypots, not 400+.
# 12+ AI skills, <2 YoE, 1 career entry, 90+ completeness.
HONEYPOT_SYNTHETIC_PROFILE_MIN_AI_SKILLS = 12
HONEYPOT_SYNTHETIC_PROFILE_MIN_COMPLETENESS = 90.0
HONEYPOT_SYNTHETIC_PROFILE_MAX_YOE = 2.0
HONEYPOT_SYNTHETIC_PROFILE_MAX_HISTORY = 1

# Technology age: stricter. Only flag if skill window is *clearly* before
# the tech existed (3+ years, not 1).
HONEYPOT_TECH_AGE_GRACE_YEARS = 3

# Skill experience contradiction: stricter. Need 5+ skills (not 3) and
# ALL with zero endorsements.
HONEYPOT_SKILL_EXP_CONTRADICTION_MIN = 5

# Technology age: skill names reference technologies that didn't exist
# in claimed start years (e.g., claiming PyTorch expertise starting 2015).
TECH_RELEASE_YEARS = {
    "pytorch": 2016,
    "tensorflow": 2015,
    "transformers": 2017,
    "huggingface": 2016,
    "hugging face": 2016,
    "langchain": 2022,
    "llamaindex": 2022,
    "pinecone": 2019,
    "weaviate": 2019,
    "qdrant": 2021,
    "chroma": 2022,
    "faiss": 2017,
    "sentence-transformers": 2019,
    "sentence transformers": 2019,
    "sbert": 2019,
    "colbert": 2020,
    "lora": 2021,
    "qlora": 2023,
    "peft": 2022,
    "mlflow": 2018,
    "kubeflow": 2017,
    "wandb": 2017,
    "weights & biases": 2017,
    "rag": 2020,
    "retrieval-augmented generation": 2020,
    "openai": 2020,
    "chatgpt": 2022,
    "gpt-4": 2023,
    "claude": 2023,
    "gemini": 2023,
    "bert": 2018,
    "gpt-3": 2020,
    "gpt-2": 2019,
    "roberta": 2019,
    "t5": 2019,
    "ragas": 2023,
    "llamaindex": 2022,
    "langgraph": 2024,
    "ollama": 2023,
    "vllm": 2023,
}

# Cross-field consistency: certain field combinations are suspicious
# (e.g., "NLP Researcher" + "consulting-only career" + "no skills in NLP").
NLP_KEYWORDS = ["nlp", "natural language", "language model", "llm", "text classification",
                "named entity", "sentiment", "transformer", "tokeniz"]
CV_KEYWORDS = ["computer vision", "image classification", "object detection",
               "segmentation", "opencv", "yolo", "cnn", "convolutional",
               "image recognition", "visual recognition", "image segmentation"]
SPEECH_KEYWORDS = ["speech recognition", "tts", "asr", "text-to-speech", "speech synthesis",
                   "wake word", "speaker diarization", "audio classification"]
ROBOTICS_KEYWORDS = ["robotics", "ros", "slam", "path planning", "robot operating system",
                     "manipulation", "autonomous vehicle", "autonomous driving"]

# ============================================================================
# Behavioral signal normalization
# ============================================================================
# Log-normalized signals use log1p to compress the long tail.
# Inverse signals flip high=bad to high=good.

BEHAVIORAL_SIGNAL_WEIGHTS = {
    "search_appearance_30d": 0.20,
    "saved_by_recruiters_30d": 0.20,
    "endorsements_received": 0.10,
    "connection_count": 0.05,
    "recruiter_response_rate": 0.15,
    "avg_response_time_hours": 0.10,  # inverse
    "interview_completion_rate": 0.05,
    "open_to_work_flag": 0.05,
    "github_activity_score": 0.05,
    "profile_completeness_score": 0.05,
}
assert abs(sum(BEHAVIORAL_SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "BEHAVIORAL_SIGNAL_WEIGHTS must sum to 1.0"

# ============================================================================
# Location fit
# ============================================================================
# JD says: Pune/Noida preferred. India bonus. US/Canada/UK partial.
LOCATION_SCORES = {
    "india": 1.0,
    "united states": 0.5,
    "usa": 0.5,
    "us": 0.5,
    "united kingdom": 0.5,
    "uk": 0.5,
    "canada": 0.5,
}
LOCATION_DEFAULT_SCORE = 0.3

# Indian city boost (Pune, Noida get +0.1)
INDIAN_CITY_BOOST = {"pune": 0.1, "noida": 0.1, "delhi ncr": 0.05, "delhi": 0.05, "gurugram": 0.05, "gurgaon": 0.05, "mumbai": 0.05, "bengaluru": 0.05, "bangalore": 0.05, "hyderabad": 0.05, "chennai": 0.05}
INDIAN_CITY_BOOST_MAX = 0.5  # cap so we don't exceed 1.0

# ============================================================================
# Notice period scoring
# ============================================================================
# ≤30 = 1.0, ≤60 = 0.7, ≤90 = 0.4, >90 = 0.2

def notice_period_score(days: int) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.7
    if days <= 90:
        return 0.4
    return 0.2

# ============================================================================
# Education tier
# ============================================================================
EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.7,
    "tier_3": 0.4,
    "tier_4": 0.1,
    "unknown": 0.3,
}
EDUCATION_DEFAULT_SCORE = 0.3

# ============================================================================
# Output constraints
# ============================================================================
TOP_K = 100
MIN_REASONING_LEN = 1
MAX_REASONING_LEN = 300  # Per spec — but be liberal; < 1 line is enough.

# ============================================================================
# JD requirements — used for evidence-based reasoning
# ============================================================================
# Anchored to the JD "what you absolutely need" list.
# These strings appear in the reasoning when matched, giving reviewers a
# way to see WHY a candidate scored high.
JD_MUST_HAVE_PATTERNS = {
    "embeddings_or_retrieval": [
        "embedding", "sentence-transformer", "sentence transformer", "sbert",
        "colbert", "vector search", "vector db", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "vespa", "information retrieval",
        "semantic search", "bm25", "retrieval augmented", "rag",
        "retrieval-augmented", "hybrid search",
    ],
    "vector_db_or_search": [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "vespa",
        "elasticsearch", "opensearch", "solr", "lucene", "vector search",
        "vector db", "vector database", "annoy", "hnsw",
    ],
    "strong_python": ["python", "pytest", "pydantic", "fastapi", "django", "flask"],
    "eval_framework": [
        "ndcg", "mrr", "map", "auc", "evaluation framework",
        "offline evaluation", "online evaluation", "a/b test", "ab test",
        "learning to rank", "learning-to-rank", "l2r",
    ],
}

# ============================================================================
# Sanity log thresholds
# ============================================================================
HONEYPOT_RATE_TARGET = 0.0  # Stage 3 disqualifies at >10%, we target 0%
TRAP_DEDUP_LOG_EVERY = 10_000  # Log every N candidates during long ops
