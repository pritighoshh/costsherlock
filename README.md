# 🔍 CostSherlock
> An agentic system for causal attribution of AWS cost anomalies

CostSherlock is a four-agent AI pipeline that automatically investigates unexpected spikes in your AWS bill, traces them to specific CloudTrail events, and produces evidence-backed explanation reports — all without requiring manual log analysis. It is designed for cloud engineers and FinOps practitioners who need to answer "why did our bill jump last Tuesday?" in minutes rather than hours, and for engineering teams that want an auditable, citation-grounded record of every cost incident.

---

## Architecture

```mermaid
flowchart LR
    CD[(Cost Explorer\nJSON export)] --> S

    subgraph Pipeline
        S[🛡️ Sentinel\nAnomaly Detection\nZ-score · 14-day window]
        D[🕵️ Detective\nCloudTrail Correlation\nProximity scoring]
        A[🔬 Analyst\nRAG Causal Reasoning\nCost math validation]
        N[📝 Narrator\nReport Generation\nInline citations]

        S -->|Anomaly objects| D
        D -->|Suspect events| A
        A -->|Hypotheses + ruled-out| N
        KB[(RAG Knowledge Base\n50 AWS pricing docs\nChromaDB + MiniLM)] --> A
    end

    N -->|Markdown reports| DB

    subgraph Dashboard ["🖥️ Streamlit Dashboard (5 views)"]
        DB[Timeline\nCost trend + anomaly table]
        DB --> V2[Investigation\nReport viewer + nav]
        DB --> V3[Evidence Explorer\nPer-hypothesis cards]
        DB --> V4[Compare\nCross-anomaly charts]
        DB --> V5[Feedback\nHuman audit + corrections]
    end
```

---

## Quick Start

### Demo Mode (no AWS account needed)

```bash
git clone https://github.com/naik-vatsal/costsherlock.git
cd costsherlock

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline on bundled synthetic data
python demo.py

# Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. Three pre-investigated anomalies (EC2, S3, CloudWatch) are loaded automatically.

### With Your Own AWS Data

**1. Export Cost Explorer data** (AWS Console → Cost Explorer → Download as CSV, or use the CLI):

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-02-28 \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output json > data/cost_exports/my_costs.json
```

**2. Export CloudTrail events** (AWS Console → CloudTrail → Event history → Download JSON):

```bash
# Place CloudTrail JSON files in:
data/cloudtrail/
```

**3. Run the pipeline:**

```bash
python pipeline.py \
  --cost data/cost_exports/my_costs.json \
  --logs data/cloudtrail/
```

**4. Launch the dashboard:**

```bash
streamlit run dashboard/app.py
```

---

## How It Works

**Sentinel** reads your AWS Cost Explorer export and applies a z-score anomaly detector with a 14-day rolling window (threshold: z ≥ 2.5). Each day's spend per service is compared against the recent baseline; dates that deviate significantly are flagged as `Anomaly` objects and passed downstream. This stage is fully deterministic — no LLM calls.

**Detective** loads your CloudTrail JSON exports and scores every event against each anomaly using a proximity function that weighs temporal closeness and event type. Only events on AWS's mutating-event whitelist (e.g. `RunInstances`, `PutBucketLifecycleConfiguration`, `ModifyDBInstance`) are considered, filtering out read-only noise. The top-scored events become `SuspectEvent` objects.

**Analyst** is the reasoning core. For each anomaly, it retrieves the eight most relevant chunks from the RAG knowledge base (50 AWS pricing and troubleshooting documents, embedded with `all-MiniLM-L6-v2` and stored in ChromaDB), then asks Claude Sonnet to evaluate every suspect event against three criteria: correct service, plausible causal mechanism, and cost-math consistency. The cost math check is quantitative — if the calculated impact is less than 20% or more than 400% of the observed delta, confidence is automatically halved. Events that fail are moved to a `ruled_out` list with a reason category (`WRONG_MECHANISM`, `WRONG_MAGNITUDE`, etc.).

> **Silent failure example:** A `PutBucketPolicy` event appears 30 minutes before an S3 cost spike. It looks suspicious — but bucket policies control *access permissions*, not storage class or pricing. The Analyst classifies this as `WRONG_MECHANISM` and rules it out, even though it is temporally correlated. Without this check, a simpler system would confidently blame the wrong event.

**Narrator** converts the Analyst's structured output into a seven-section markdown report (`Executive Summary`, `Root Cause Analysis`, `Cost Breakdown`, `Evidence Chain`, `Ruled Out`, `Remediation`, `Confidence & Caveats`). Every factual claim is required to carry an inline citation tag — `[CloudTrail: RunInstances]`, `[Pricing: ec2_ondemand_pricing.md]`, or `[Metric: cost delta]`. Conclusions that go beyond the evidence are explicitly labelled `[INFERENCE]`. A post-processing pass enforces this: any line containing a cost figure, causal claim, timestamp, or actor attribution that lacks a citation is automatically tagged.

**The Dashboard** loads reports from the pipeline and presents them across five views. The Timeline view shows a cost trend chart and an anomaly severity table with one-click investigation. The Investigation view renders the full report with Previous/Next navigation across all investigated anomalies. The Evidence Explorer breaks down each hypothesis and its supporting evidence into expandable cards. The Compare view shows a grouped bar chart of cost deltas across anomalies. The Feedback view lets engineers rate each report as actionable or not, and enter the actual root cause if the system missed it — corrections are recorded as JSON and can be ingested back into the RAG store to improve future investigations.

---

## Dashboard

> **Screenshots coming soon** — actual dashboard screenshots will be added after the demo recording.

| View | Description |
|------|-------------|
| **Timeline** | Interactive cost trend chart (Plotly), 4 KPI cards (total anomalies, highest severity, total excess spend, services affected), sortable anomaly table with one-click "Investigate" buttons |
| **Investigation** | Full 7-section report rendered in markdown, Previous/Next navigation across all investigated anomalies, confidence colour coding, status banner |
| **Evidence Explorer** | Per-hypothesis expandable cards showing evidence items, cost calculation, causal mechanism, and ruled-out events |
| **Compare** | Grouped bar chart comparing expected vs. actual costs across all anomalies; side-by-side confidence scores |
| **Feedback** | Actionability rating (radio), free-text actual root cause field, submission saves a timestamped JSON file for the feedback loop |

---

## Evaluation Results

Evaluated on 3 synthetic anomalies (EC2 compute overprovisioning, S3 lifecycle misconfiguration, CloudWatch logging misconfiguration) with ground-truth labels and known causal CloudTrail events.

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Causal Attribution Accuracy | ≥ 80% | **100.0%** (3/3) | ✅ PASS |
| Evidence Recall | ≥ 85% | **100.0%** | ✅ PASS |
| Faithfulness Score (citation ratio) | ≥ 90% | **93.0%** | ✅ PASS |
| Time to Explanation | ≤ 300 s | **32.8 s avg** | ✅ PASS |
| Human Audit Pass Rate | ≥ 70% | pending feedback | ⏳ manual |
| Time to Insight (UX) | < 3 min | pending user study | ⏳ manual |
| Feedback Loop Quality | methodology | 0 corrections ingested | ⏳ manual |

**Faithfulness** is measured as the fraction of content lines in a report that carry an inline citation or `[INFERENCE]` tag. Table rows, section headers, and meta-commentary bullets are excluded from the denominator as non-claims.

**Causal Attribution Accuracy** uses a three-level fuzzy match: exact string → synonym group (e.g. `monitoring_overprovisioning` maps to `logging_misconfiguration`) → shared keyword after stopword removal.

**Human Audit Pass Rate** and **Time to Insight** require manual evaluation sessions; methodology is implemented in `evaluation/metrics.py`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM | Anthropic Claude Sonnet (`claude-sonnet-4-20250514`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB (local, persistent) |
| RAG Framework | LangChain |
| Evaluation | RAGAS (with citation-ratio fallback) |
| Frontend | Streamlit |
| Charting | Plotly |
| AWS SDK | boto3 |
| Data | pandas, numpy |
| Retry logic | tenacity |
| Terminal output | Rich |

---

## Project Structure

```
costsherlock/
├── agents/
│   ├── __init__.py          # Pydantic models: Anomaly, SuspectEvent, Hypothesis, InvestigationReport
│   ├── sentinel.py          # Agent 1: z-score anomaly detection on cost time series
│   ├── detective.py         # Agent 2: CloudTrail event proximity scoring
│   ├── analyst.py           # Agent 3: RAG-powered causal reasoning + cost math validation
│   └── narrator.py          # Agent 4: cited markdown report generation
├── rag/
│   ├── ingest.py            # Document chunking (500 tok / 50 overlap) + embedding pipeline
│   ├── retriever.py         # ChromaDB query interface (top-k semantic search)
│   └── documents/           # 50 AWS pricing + troubleshooting markdown documents
│       ├── ec2_ondemand_pricing.md
│       ├── s3_storage_classes.md
│       ├── cost_trap_debug_logging.md
│       └── ...              # (48 more)
├── dashboard/
│   ├── app.py               # Streamlit app — 5 views, ~1600 lines
│   ├── components/          # Reusable Streamlit component helpers
│   └── assets/              # Static assets (CSS, images)
├── data/
│   ├── cost_exports/        # Place Cost Explorer JSON exports here
│   ├── cloudtrail/          # Place CloudTrail JSON exports here
│   ├── synthetic/
│   │   ├── demo_cost.json         # 225-day synthetic cost time series (5 services)
│   │   ├── demo_cloudtrail/       # 34 synthetic CloudTrail events across 5 files
│   │   └── ground_truth.json      # 3 labelled anomalies with root cause + causal event
│   └── feedback/            # Human feedback JSON files (written by dashboard)
├── evaluation/
│   ├── metrics.py           # All 7 evaluation metrics
│   ├── run_eval.py          # Pipeline runner + Rich results table
│   └── results.json         # Latest evaluation run output
├── pipeline.py              # Orchestrator: chains all 4 agents, saves reports
├── demo.py                  # Demo mode entry point (uses synthetic data)
├── requirements.txt
├── .env                     # ANTHROPIC_API_KEY (never committed)
└── CLAUDE.md                # Architecture and coding guidelines
```

---

## Team

| Member | Contributions |
|--------|--------------|
| **Vatsal Naik** | Backend pipeline (Agents 1–4), RAG knowledge base (50 documents, ChromaDB ingestion), evaluation framework (7 metrics, RAGAS integration), pipeline orchestrator, demo mode |
| **Priti Ghosh** | Streamlit dashboard (5 views, ~1600 lines), Plotly visualizations, feedback mechanism, usability evaluation methodology |

---

## Cost

All LLM costs are incurred via the Anthropic API. Embeddings and vector search run entirely locally at zero API cost.

| Item | Details | Estimated Cost |
|------|---------|---------------|
| Agent development & iteration | ~200 LLM calls during build (Analyst + Narrator prompts) | ~$2.00 |
| RAG knowledge base ingestion | Local embeddings only | $0.00 |
| Evaluation runs | 6 LLM calls × ~4 runs | ~$1.20 |
| Demo pipeline (3 anomalies) | 6 LLM calls @ ~$0.08/run | ~$0.08 |
| Dashboard development | Streamlit only, no LLM | $0.00 |
| **Total** | | **< $10** |

> Per-investigation cost at runtime: ~$0.08 USD for 3 anomalies (6 LLM calls, ~4K tokens each).

---

## Limitations & Future Work

- **Real-time API polling** — the current system ingests batch JSON exports. A production version would stream from the Cost Explorer and CloudTrail APIs on a schedule, eliminating the manual export step.
- **Multi-cloud support** — the architecture is cloud-agnostic in principle; the Detective and Analyst agents would need adapters for GCP Cloud Audit Logs and Azure Monitor.
- **Automated remediation** — currently the Narrator suggests remediation steps as prose. Future work: generate and optionally apply AWS CLI / Terraform patches with human approval.
- **Fine-tuned anomaly classifier** — the z-score detector is a strong baseline but can produce false positives on services with high natural variance (e.g. data transfer). A learned classifier trained on historical labelled anomalies would improve precision.
- **Organizational knowledge integration** — Tier 3 data (team-specific runbooks, internal incident history, reserved instance inventory) would make causal reasoning significantly more precise. The RAG store is designed to accept additional documents at any time.
- **Multi-anomaly correlation** — the current pipeline investigates each anomaly independently. Correlated anomalies across services (e.g. a deployment that simultaneously spikes EC2 and data transfer) are not yet linked.
