# CostSherlock - CLAUDE.md

## Project Overview
CostSherlock is a multi-agent system that investigates AWS cost anomalies and generates evidence-backed explanation reports. It has a 4-agent Python backend + a Streamlit dashboard frontend.

## Architecture
```
Cost Explorer JSON → Sentinel (anomaly detection) → Detective (CloudTrail correlation)
→ Analyst (RAG reasoning via ChromaDB) → Narrator (report generation) → Streamlit Dashboard
```

## Tech Stack
- **Language:** Python 3.11+
- **LLM:** Anthropic Claude Sonnet (claude-sonnet-4-20250514) via API
- **Embeddings:** sentence-transformers all-MiniLM-L6-v2 (local, FREE — no API needed)
- **Vector DB:** ChromaDB (local, persistent)
- **RAG Framework:** LangChain
- **Evaluation:** RAGAS
- **Frontend:** Streamlit
- **AWS SDK:** boto3
- **Data:** pandas, numpy

## Project Structure
```
costsherlock/
├── agents/
│   ├── __init__.py
│   ├── sentinel.py      # Agent 1: z-score anomaly detection
│   ├── detective.py     # Agent 2: CloudTrail event correlation
│   ├── analyst.py       # Agent 3: RAG causal reasoning
│   └── narrator.py      # Agent 4: Report generation with citations
├── rag/
│   ├── __init__.py
│   ├── ingest.py        # Document chunking & embedding pipeline
│   ├── retriever.py     # ChromaDB query interface
│   └── documents/       # AWS pricing docs (markdown/text files)
├── dashboard/
│   ├── app.py           # Streamlit main app
│   ├── components/      # Reusable Streamlit components
│   └── assets/          # Static assets (CSS, images)
├── data/
│   ├── cost_exports/    # Cost Explorer JSON exports
│   ├── cloudtrail/      # CloudTrail log JSON exports
│   └── synthetic/       # Injected anomaly ground truth
├── evaluation/
│   ├── metrics.py       # RAGAS + custom evaluation metrics
│   ├── test_attribution.py  # Causal accuracy test suite
│   └── ground_truth.json
├── pipeline.py          # Orchestrator: chains all 4 agents
├── demo.py              # Demo mode with bundled synthetic data
├── requirements.txt
├── .env                 # API keys (never commit)
├── .gitignore
├── CLAUDE.md            # This file
└── README.md
```

## Build Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run pipeline on sample data
python pipeline.py --data data/synthetic/demo_cost.json --logs data/synthetic/demo_cloudtrail/

# Run Streamlit dashboard
streamlit run dashboard/app.py

# Run evaluation suite
python -m pytest evaluation/ -v

# Run demo mode (no AWS account needed)
python demo.py
```

## Code Style
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- f-strings for string formatting
- dataclasses or Pydantic models for structured data between agents
- All inter-agent communication uses JSON-serializable dataclasses
- Error handling: wrap all LLM API calls in try/except with retries

## Key Design Decisions
- MVP uses batch JSON file ingestion, NOT live AWS API polling
- ChromaDB runs locally (no cloud vector DB dependency)
- Embeddings use local sentence-transformers (all-MiniLM-L6-v2) — zero API cost
- LLM calls use Anthropic Claude Sonnet (claude-sonnet-4-20250514) via anthropic Python SDK
- Agent 3 (Analyst) must validate causal plausibility quantitatively, not just temporally
- Every claim in Narrator output must have a citation or be labeled [INFERENCE]
- Streamlit chosen over React to keep entire stack in Python

## Inter-Agent Data Contract
All agents communicate via these Pydantic models defined in agents/__init__.py:
- `Anomaly`: service, date, cost, expected_cost, z_score, delta
- `SuspectEvent`: event_name, event_time, user_arn, resource_arn, proximity_score, summary
- `Hypothesis`: rank, root_cause, confidence, evidence_list, cost_calculation, causal_mechanism
- `InvestigationReport`: anomaly, hypotheses, ruled_out, remediation, overall_confidence

## Testing
- Use pytest for all tests
- Synthetic anomaly data in data/synthetic/ serves as integration test fixtures
- RAGAS evaluation requires: question, answer, contexts fields in Dataset format

## Important Constraints
- NEVER commit .env or API keys
- NEVER use print() for logging — use Python logging module
- AWS CloudTrail MUTATING_EVENTS whitelist: RunInstances, TerminateInstances, ModifyDBInstance, PutBucketLifecycleConfiguration, PutBucketPolicy, CreateFunction20150331, UpdateFunctionConfiguration, CreateNatGateway, ModifyInstanceAttribute, CreateAutoScalingGroup
- RAG chunk size: 500 tokens, 50 token overlap
- Z-score anomaly threshold: 2.5 on 14-day rolling window
