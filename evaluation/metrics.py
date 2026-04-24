"""CostSherlock evaluation metrics — all 7 from the project proposal.

Metrics
-------
Backend (automated):
  1. causal_attribution_accuracy  — top hypothesis category vs ground truth
  2. evidence_recall              — GT events found in evidence strings
  3. faithfulness_score           — citation ratio (RAGAS if available)
  4. time_to_explanation          — elapsed_seconds per report
  5. human_audit_pass_rate        — feedback JSON "actionable" rate

Frontend (manual / descriptive):
  6. time_to_insight              — description of manual UX test
  7. feedback_loop_quality        — methodology description
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Targets (from proposal) ───────────────────────────────────────────────────
TARGET_CAUSAL_ACCURACY: float = 0.80        # ≥ 80%
TARGET_EVIDENCE_RECALL: float = 0.85        # ≥ 85%
TARGET_FAITHFULNESS: float = 0.90           # ≥ 90%
TARGET_TIME_TO_EXPLANATION: float = 300.0   # ≤ 300 s
TARGET_HUMAN_AUDIT_PASS_RATE: float = 0.70  # ≥ 70%


# ─────────────────────────────────────────────────────────────────────────────
# 1. Causal Attribution Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def causal_attribution_accuracy(
    reports: list,
    ground_truth_path: str | Path,
) -> dict[str, Any]:
    """Compare each report's top-hypothesis category against ground truth.

    Args:
        reports: List of ``InvestigationReport`` objects from the pipeline.
        ground_truth_path: Path to the ground-truth JSON file
            (list of ``{anomaly_service, anomaly_date, root_cause_category, …}``).

    Returns:
        ``{accuracy, correct_count, total, per_anomaly: [{service, date,
        expected, actual, correct}]}``
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.warning("Ground truth file not found: %s", gt_path)
        return {"error": f"Ground truth file not found: {gt_path}"}

    with gt_path.open(encoding="utf-8") as fh:
        ground_truth: list[dict] = json.load(fh)

    # Index GT by (service, date)
    gt_index: dict[tuple[str, str], dict] = {
        (entry["anomaly_service"], entry["anomaly_date"]): entry
        for entry in ground_truth
    }

    per_anomaly: list[dict] = []
    correct_count = 0

    for report in reports:
        key = (report.anomaly.service, report.anomaly.date)
        gt_entry = gt_index.get(key)
        if gt_entry is None:
            logger.debug("No GT entry for %s / %s — skipping", *key)
            continue

        expected = gt_entry["root_cause_category"].lower().replace("-", "_")
        top_hyp = report.hypotheses[0] if report.hypotheses else None
        actual = top_hyp.category.lower().replace("-", "_").replace(" ", "_") if top_hyp else "no_hypothesis"

        correct = _categories_match(expected, actual)
        if correct:
            correct_count += 1

        per_anomaly.append(
            {
                "service": report.anomaly.service,
                "date": report.anomaly.date,
                "expected": expected,
                "actual": actual,
                "correct": correct,
            }
        )

    total = len(per_anomaly)
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "correct_count": correct_count,
        "total": total,
        "per_anomaly": per_anomaly,
        "target": TARGET_CAUSAL_ACCURACY,
        "pass": accuracy >= TARGET_CAUSAL_ACCURACY,
    }


# Synonym groups: any two categories in the same group are considered equivalent.
_CATEGORY_SYNONYMS: list[frozenset[str]] = [
    frozenset({"logging_misconfiguration", "monitoring_misconfiguration",
               "monitoring_overprovisioning", "cloudwatch_misconfiguration",
               "log_misconfiguration", "cloudtrail_misconfiguration"}),
    frozenset({"storage_misconfiguration", "storage_class_misconfiguration",
               "s3_misconfiguration", "bucket_misconfiguration"}),
    frozenset({"compute_overprovisioning", "ec2_overprovisioning",
               "instance_overprovisioning", "compute_misconfiguration"}),
    frozenset({"network_misconfiguration", "data_transfer_spike",
               "nat_misconfiguration", "vpc_misconfiguration"}),
    frozenset({"database_misconfiguration", "rds_misconfiguration",
               "db_misconfiguration"}),
    frozenset({"lambda_misconfiguration", "serverless_misconfiguration"}),
]


def _categories_match(expected: str, actual: str) -> bool:
    """Fuzzy category match: exact, synonym group, substring, or shared keyword."""
    if expected == actual:
        return True
    # Synonym group match
    for group in _CATEGORY_SYNONYMS:
        if expected in group and actual in group:
            return True
    # Substring in either direction
    if expected in actual or actual in expected:
        return True
    # Shared keyword (split on underscores)
    stopwords = {"aws", "amazon", "the", "a", "over", "mis"}
    exp_words = set(expected.split("_")) - stopwords
    act_words = set(actual.split("_")) - stopwords
    return bool(exp_words & act_words)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evidence Recall
# ─────────────────────────────────────────────────────────────────────────────

def evidence_recall(
    reports: list,
    ground_truth_path: str | Path,
) -> dict[str, Any]:
    """Fraction of ground-truth causal events found in each report's evidence.

    Each GT entry has one ``root_cause_event`` (a CloudTrail event name).
    Recall per anomaly is 1.0 if that event name appears anywhere in the
    evidence strings or report markdown, 0.0 otherwise.

    Args:
        reports: List of ``InvestigationReport`` objects.
        ground_truth_path: Path to ground-truth JSON.

    Returns:
        ``{average_recall, per_anomaly: [{service, date, gt_event,
        found, recall}]}``
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        return {"error": f"Ground truth file not found: {gt_path}"}

    with gt_path.open(encoding="utf-8") as fh:
        ground_truth: list[dict] = json.load(fh)

    gt_index = {
        (e["anomaly_service"], e["anomaly_date"]): e for e in ground_truth
    }

    per_anomaly: list[dict] = []

    for report in reports:
        key = (report.anomaly.service, report.anomaly.date)
        gt_entry = gt_index.get(key)
        if gt_entry is None:
            continue

        gt_event = gt_entry["root_cause_event"]  # e.g. "RunInstances"

        # Search all evidence strings + full report markdown
        search_corpus = report.report_markdown
        for hyp in report.hypotheses:
            search_corpus += " ".join(hyp.evidence)
            search_corpus += hyp.root_cause
            search_corpus += hyp.causal_mechanism

        found = gt_event.lower() in search_corpus.lower()

        per_anomaly.append(
            {
                "service": report.anomaly.service,
                "date": report.anomaly.date,
                "gt_event": gt_event,
                "found": found,
                "recall": 1.0 if found else 0.0,
            }
        )

    average_recall = (
        sum(r["recall"] for r in per_anomaly) / len(per_anomaly)
        if per_anomaly else 0.0
    )

    return {
        "average_recall": round(average_recall, 4),
        "per_anomaly": per_anomaly,
        "target": TARGET_EVIDENCE_RECALL,
        "pass": average_recall >= TARGET_EVIDENCE_RECALL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Faithfulness Score
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"\[(CloudTrail|Pricing|Metric|RAG|Source|Ref):", re.IGNORECASE
)
_INFERENCE_RE = re.compile(r"\[INFERENCE\]", re.IGNORECASE)

# Lines that are structural (headers, blank, dividers, table rows, meta-commentary)
# — not factual claims needing grounding.
_STRUCTURAL_RE = re.compile(
    r"^\s*("
    r"#{1,6}\s"              # markdown headers  ## ...
    r"|[-*]{3,}"             # horizontal rules  ---
    r"|\|[-| :]+\|?"         # table divider rows  |---|---|
    r"|\|"                   # table content rows  | foo | bar |
    r"|>\s*$"                # block quotes
    r"|$"                    # empty lines
    r"|- \*\*(?:Confident|Uncertain|Would increase confidence)"  # caveats bullets
    r"|\*\*[^*]{1,60}:\*\*\s*$"  # standalone bold label lines  **Label:**
    r"|[^.!?]{1,80}:\s*$"        # short label-only lines ending with colon (transitions)
    r")"
)


def faithfulness_score(
    report_markdown: str,
    retrieved_contexts: list[str] | None = None,
) -> dict[str, Any]:
    """Measure how well the report's claims are grounded in cited evidence.

    Tries RAGAS ``faithfulness`` metric first.  Falls back to a
    citation-ratio heuristic if RAGAS is unavailable or the contexts list
    is empty.

    Citation ratio = (cited lines + inference-labelled lines) / content lines.

    Args:
        report_markdown: Full markdown text of one investigation report.
        retrieved_contexts: RAG context chunks used during generation.
            Required for RAGAS; optional for the fallback.

    Returns:
        ``{score, method, target, pass}``
    """
    # ── Attempt RAGAS ────────────────────────────────────────────────────────
    if retrieved_contexts:
        try:
            score = _ragas_faithfulness(report_markdown, retrieved_contexts)
            return {
                "score": round(score, 4),
                "method": "ragas",
                "target": TARGET_FAITHFULNESS,
                "pass": score >= TARGET_FAITHFULNESS,
            }
        except Exception as exc:
            logger.info("RAGAS unavailable (%s) — using citation-ratio fallback", exc)

    # ── Citation-ratio fallback ───────────────────────────────────────────────
    score = _citation_ratio(report_markdown)
    return {
        "score": round(score, 4),
        "method": "citation_ratio",
        "target": TARGET_FAITHFULNESS,
        "pass": score >= TARGET_FAITHFULNESS,
    }


def _ragas_faithfulness(report_markdown: str, contexts: list[str]) -> float:
    """Run RAGAS faithfulness evaluation."""
    from datasets import Dataset  # type: ignore[import]
    from ragas import evaluate  # type: ignore[import]
    from ragas.metrics import faithfulness  # type: ignore[import]

    data = Dataset.from_dict(
        {
            "question": ["What caused this AWS cost anomaly?"],
            "answer": [report_markdown],
            "contexts": [contexts],
        }
    )
    result = evaluate(data, metrics=[faithfulness])
    return float(result["faithfulness"])


def _citation_ratio(report_markdown: str) -> float:
    """Heuristic: fraction of content lines with a citation or [INFERENCE] tag."""
    lines = report_markdown.splitlines()
    content_lines = [
        ln for ln in lines if not _STRUCTURAL_RE.match(ln)
    ]
    if not content_lines:
        return 0.0
    cited = sum(
        1 for ln in content_lines
        if _CITATION_RE.search(ln) or _INFERENCE_RE.search(ln)
    )
    return cited / len(content_lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time to Explanation
# ─────────────────────────────────────────────────────────────────────────────

def time_to_explanation(reports: list) -> dict[str, Any]:
    """Summarise pipeline elapsed time per investigation report.

    Args:
        reports: List of ``InvestigationReport`` objects.

    Returns:
        ``{per_anomaly: [{service, date, elapsed_seconds}],
        average_seconds, max_seconds, target, pass}``
    """
    if not reports:
        return {
            "per_anomaly": [],
            "average_seconds": 0.0,
            "max_seconds": 0.0,
            "target": TARGET_TIME_TO_EXPLANATION,
            "pass": True,
        }

    per_anomaly = [
        {
            "service": r.anomaly.service,
            "date": r.anomaly.date,
            "elapsed_seconds": round(r.elapsed_seconds, 2),
        }
        for r in reports
    ]
    times = [r.elapsed_seconds for r in reports]
    avg = sum(times) / len(times)
    maximum = max(times)

    return {
        "per_anomaly": per_anomaly,
        "average_seconds": round(avg, 2),
        "max_seconds": round(maximum, 2),
        "target": TARGET_TIME_TO_EXPLANATION,
        "pass": avg <= TARGET_TIME_TO_EXPLANATION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Human Audit Pass Rate
# ─────────────────────────────────────────────────────────────────────────────

def human_audit_pass_rate(feedback_dir: str | Path) -> dict[str, Any]:
    """Compute fraction of investigations rated "Report is actionable".

    Reads every ``*.json`` file under *feedback_dir*.  Looks for the
    ``overall`` key written by the Feedback view in the dashboard.

    Args:
        feedback_dir: Directory containing feedback JSON files.

    Returns:
        ``{pass_rate, actionable_count, total_feedback, target, pass,
        note}``
    """
    feedback_path = Path(feedback_dir)
    if not feedback_path.exists():
        return {
            "pass_rate": 0.0,
            "actionable_count": 0,
            "total_feedback": 0,
            "target": TARGET_HUMAN_AUDIT_PASS_RATE,
            "pass": False,
            "note": "No feedback directory found — submit feedback via the dashboard.",
        }

    files = list(feedback_path.glob("*.json"))
    if not files:
        return {
            "pass_rate": 0.0,
            "actionable_count": 0,
            "total_feedback": 0,
            "target": TARGET_HUMAN_AUDIT_PASS_RATE,
            "pass": False,
            "note": "No feedback files yet — expected until users submit ratings.",
        }

    actionable = 0
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if "actionable" in data.get("overall", "").lower():
                actionable += 1
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read feedback file %s: %s", fp, exc)

    total = len(files)
    rate = actionable / total if total > 0 else 0.0
    return {
        "pass_rate": round(rate, 4),
        "actionable_count": actionable,
        "total_feedback": total,
        "target": TARGET_HUMAN_AUDIT_PASS_RATE,
        "pass": rate >= TARGET_HUMAN_AUDIT_PASS_RATE,
        "note": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Time to Insight  (manual / descriptive)
# ─────────────────────────────────────────────────────────────────────────────

def time_to_insight() -> str:
    """Return the evaluation methodology for the time-to-insight UX metric.

    This is a manual test — not automated.

    Returns:
        Human-readable description of the test protocol and target.
    """
    return (
        "Manual UX test: a non-engineer is given the CostSherlock dashboard "
        "and asked to identify the root cause of the highest-severity anomaly. "
        "The time from dashboard open to verbal root-cause identification is recorded. "
        "Target: < 3 minutes (180 s). "
        "Run at least 5 participants and report median time."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feedback Loop Quality  (methodology / descriptive)
# ─────────────────────────────────────────────────────────────────────────────

def feedback_loop_quality(feedback_dir: str | Path) -> dict[str, Any]:
    """Describe the feedback-loop evaluation methodology and report feedback counts.

    Full automation requires multiple pipeline runs with and without
    human-corrected ground truth injected into the RAG knowledge base.
    This function returns the methodology and a count of available feedback.

    Args:
        feedback_dir: Directory containing feedback JSON files.

    Returns:
        ``{methodology, feedback_count, corrections_available,
        status}``
    """
    feedback_path = Path(feedback_dir)
    files = list(feedback_path.glob("*.json")) if feedback_path.exists() else []

    corrections = 0
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if data.get("actual_root_cause", "").strip():
                corrections += 1
        except (json.JSONDecodeError, OSError):
            pass

    methodology = (
        "1. Run the pipeline on synthetic data and record baseline accuracy. "
        "2. Collect human feedback corrections via the dashboard Feedback view. "
        "3. Ingest corrections as new documents into the ChromaDB RAG store. "
        "4. Re-run the pipeline on the same anomalies. "
        "5. Measure accuracy delta between run 1 and run 2. "
        "6. A positive delta indicates the feedback loop is improving quality."
    )

    status = (
        f"{corrections} correction(s) available from {len(files)} feedback file(s). "
        "Submit feedback via the dashboard to enable loop-quality measurement."
        if files
        else "No feedback collected yet — submit ratings via the dashboard Feedback view."
    )

    return {
        "methodology": methodology,
        "feedback_count": len(files),
        "corrections_available": corrections,
        "status": status,
    }
