"""CostSherlock evaluation runner.

Runs the full 4-agent pipeline on synthetic data, evaluates all 7 metrics,
prints a Rich results table, and saves results to evaluation/results.json.

Usage::

    python -m evaluation.run_eval
    python -m evaluation.run_eval --skip-pipeline   # re-use cached reports
    python -m evaluation.run_eval --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env", override=True)

from agents import InvestigationReport  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    causal_attribution_accuracy,
    evidence_recall,
    faithfulness_score,
    feedback_loop_quality,
    human_audit_pass_rate,
    time_to_explanation,
    time_to_insight,
)
from pipeline import CostSherlockPipeline  # noqa: E402

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
COST_PATH = str(_ROOT / "data" / "synthetic" / "demo_cost.json")
CLOUDTRAIL_DIR = str(_ROOT / "data" / "synthetic" / "demo_cloudtrail")
GT_PATH = str(_ROOT / "data" / "synthetic" / "ground_truth.json")
FEEDBACK_DIR = str(_ROOT / "data" / "feedback")
RESULTS_PATH = _ROOT / "evaluation" / "results.json"
REPORTS_DIR = str(_ROOT / "reports" / "eval")

# Rough cost model for Sonnet (claude-sonnet-4-6)
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000
_CHARS_PER_TOKEN = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_cost(reports: list[InvestigationReport]) -> float:
    """Rough API cost estimate based on report markdown character counts."""
    total = 0.0
    for r in reports:
        # Narrator output  ← report_markdown
        out_tokens = len(r.report_markdown) / _CHARS_PER_TOKEN
        # Analyst input heuristic: ~3× the report length
        in_tokens = out_tokens * 3
        total += (
            in_tokens * _INPUT_COST_PER_TOKEN
            + out_tokens * _OUTPUT_COST_PER_TOKEN
        )
    return total


def _pass_icon(passed: bool | None) -> str:
    if passed is None:
        return "—"
    return "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"


def _fmt_target(metric_name: str, value: float | str) -> str:
    thresholds = {
        "Causal Attribution Accuracy": "≥ 80%",
        "Evidence Recall":             "≥ 85%",
        "Faithfulness Score":          "≥ 90%",
        "Time to Explanation":         "≤ 300 s",
        "Human Audit Pass Rate":       "≥ 70%",
        "Time to Insight":             "< 3 min (manual)",
        "Feedback Loop Quality":       "manual",
    }
    return thresholds.get(metric_name, str(value))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> list[InvestigationReport]:
    """Execute the CostSherlock pipeline and return reports."""
    logger.info("Running CostSherlock pipeline on synthetic data…")
    pipeline = CostSherlockPipeline(output_dir=REPORTS_DIR)
    reports = pipeline.investigate(
        cost_data_path=COST_PATH,
        cloudtrail_dir=CLOUDTRAIL_DIR,
        output_subdir="",
    )
    logger.info("Pipeline complete: %d report(s) generated.", len(reports))
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(reports: list[InvestigationReport]) -> dict:
    """Compute all 7 metrics and return a structured results dict."""
    results: dict = {"metrics": {}, "meta": {}}

    # 1. Causal attribution accuracy
    logger.info("Computing metric 1/7: causal_attribution_accuracy…")
    results["metrics"]["causal_attribution_accuracy"] = causal_attribution_accuracy(
        reports, GT_PATH
    )

    # 2. Evidence recall
    logger.info("Computing metric 2/7: evidence_recall…")
    results["metrics"]["evidence_recall"] = evidence_recall(reports, GT_PATH)

    # 3. Faithfulness score — per-report, then average
    logger.info("Computing metric 3/7: faithfulness_score…")
    faith_scores: list[float] = []
    faith_details: list[dict] = []
    for r in reports:
        fs = faithfulness_score(r.report_markdown, retrieved_contexts=None)
        faith_scores.append(fs["score"])
        faith_details.append(
            {
                "service": r.anomaly.service,
                "date": r.anomaly.date,
                "score": fs["score"],
                "method": fs["method"],
            }
        )
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    from evaluation.metrics import TARGET_FAITHFULNESS
    results["metrics"]["faithfulness_score"] = {
        "average_score": round(avg_faith, 4),
        "method": faith_details[0]["method"] if faith_details else "n/a",
        "per_report": faith_details,
        "target": TARGET_FAITHFULNESS,
        "pass": avg_faith >= TARGET_FAITHFULNESS,
    }

    # 4. Time to explanation
    logger.info("Computing metric 4/7: time_to_explanation…")
    results["metrics"]["time_to_explanation"] = time_to_explanation(reports)

    # 5. Human audit pass rate
    logger.info("Computing metric 5/7: human_audit_pass_rate…")
    results["metrics"]["human_audit_pass_rate"] = human_audit_pass_rate(FEEDBACK_DIR)

    # 6. Time to insight (descriptive)
    logger.info("Computing metric 6/7: time_to_insight…")
    results["metrics"]["time_to_insight"] = {
        "description": time_to_insight(),
        "target": "< 3 minutes (manual test)",
        "pass": None,
    }

    # 7. Feedback loop quality (descriptive)
    logger.info("Computing metric 7/7: feedback_loop_quality…")
    results["metrics"]["feedback_loop_quality"] = feedback_loop_quality(FEEDBACK_DIR)
    results["metrics"]["feedback_loop_quality"]["pass"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rich output
# ─────────────────────────────────────────────────────────────────────────────

def _print_rich_table(results: dict, reports: list[InvestigationReport]) -> None:
    """Print a formatted Rich results table to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        _print_plain_table(results, reports)
        return

    console = Console()
    m = results["metrics"]

    table = Table(
        title="CostSherlock Evaluation Results",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
    )
    table.add_column("Metric",  style="bold", min_width=32)
    table.add_column("Target",  justify="center", min_width=20)
    table.add_column("Actual",  justify="center", min_width=20)
    table.add_column("Pass/Fail", justify="center", min_width=12)

    def _row(name: str, target: str, actual: str, passed: bool | None) -> None:
        icon = _pass_icon(passed)
        table.add_row(name, target, actual, icon)

    # 1. Causal attribution accuracy
    caa = m["causal_attribution_accuracy"]
    acc_pct = f"{caa.get('accuracy', 0) * 100:.1f}%  ({caa.get('correct_count', 0)}/{caa.get('total', 0)})"
    _row("Causal Attribution Accuracy", "≥ 80%", acc_pct, caa.get("pass"))

    # 2. Evidence recall
    er = m["evidence_recall"]
    recall_pct = f"{er.get('average_recall', 0) * 100:.1f}%"
    _row("Evidence Recall", "≥ 85%", recall_pct, er.get("pass"))

    # 3. Faithfulness
    fs = m["faithfulness_score"]
    faith_pct = f"{fs.get('average_score', 0) * 100:.1f}%  [{fs.get('method', '')}]"
    _row("Faithfulness Score", "≥ 90%", faith_pct, fs.get("pass"))

    # 4. Time to explanation
    tte = m["time_to_explanation"]
    time_str = f"avg {tte.get('average_seconds', 0):.1f}s  max {tte.get('max_seconds', 0):.1f}s"
    _row("Time to Explanation", "≤ 300 s", time_str, tte.get("pass"))

    # 5. Human audit pass rate
    hapr = m["human_audit_pass_rate"]
    if hapr.get("total_feedback", 0) == 0:
        hapr_str = "0 feedback files"
        hapr_pass: bool | None = None
    else:
        hapr_str = f"{hapr.get('pass_rate', 0) * 100:.1f}%  ({hapr.get('actionable_count', 0)}/{hapr.get('total_feedback', 0)})"
        hapr_pass = hapr.get("pass")
    _row("Human Audit Pass Rate", "≥ 70%", hapr_str, hapr_pass)

    # 6. Time to insight
    _row("Time to Insight", "< 3 min", "manual test", None)

    # 7. Feedback loop quality
    flq = m["feedback_loop_quality"]
    flq_str = f"{flq.get('corrections_available', 0)} correction(s)"
    _row("Feedback Loop Quality", "manual", flq_str, None)

    console.print()
    console.print(table)
    console.print()

    # ── Per-anomaly breakdown ────────────────────────────────────────────────
    detail = Table(
        title="Per-Anomaly Detail",
        box=box.SIMPLE_HEAVY,
        title_style="bold yellow",
        header_style="bold yellow",
        show_lines=False,
    )
    detail.add_column("Service",    min_width=22)
    detail.add_column("Date",       min_width=12)
    detail.add_column("Category ✓?", justify="center", min_width=14)
    detail.add_column("Evidence ✓?", justify="center", min_width=14)
    detail.add_column("Faithfulness", justify="center", min_width=14)
    detail.add_column("Time (s)",    justify="right",   min_width=10)

    # Build lookup dicts keyed by (service, date)
    cat_lookup = {
        (e["service"], e["date"]): e
        for e in caa.get("per_anomaly", [])
    }
    ev_lookup = {
        (e["service"], e["date"]): e
        for e in er.get("per_anomaly", [])
    }
    faith_lookup = {
        (e["service"], e["date"]): e
        for e in fs.get("per_report", [])
    }
    time_lookup = {
        (e["service"], e["date"]): e
        for e in tte.get("per_anomaly", [])
    }

    for r in reports:
        key = (r.anomaly.service, r.anomaly.date)
        cat_e  = cat_lookup.get(key, {})
        ev_e   = ev_lookup.get(key, {})
        faith_e = faith_lookup.get(key, {})
        time_e  = time_lookup.get(key, {})

        cat_icon  = "[green]✓[/green]" if cat_e.get("correct") else "[red]✗[/red]"
        ev_icon   = "[green]✓[/green]" if ev_e.get("found")   else "[red]✗[/red]"
        faith_val = f"{faith_e.get('score', 0) * 100:.0f}%"
        time_val  = f"{time_e.get('elapsed_seconds', 0):.1f}"

        detail.add_row(
            r.anomaly.service,
            r.anomaly.date,
            cat_icon,
            ev_icon,
            faith_val,
            time_val,
        )

    console.print(detail)
    console.print()

    # ── Cost estimate ────────────────────────────────────────────────────────
    cost_usd = _estimate_cost(reports)
    api_calls = len(reports) * 2  # Analyst + Narrator per anomaly
    console.print(
        f"[dim]Pipeline: {len(reports)} anomalies  ·  "
        f"~{api_calls} LLM API calls  ·  "
        f"estimated cost [bold]${cost_usd:.4f} USD[/bold][/dim]"
    )
    console.print(
        f"[dim]Results saved → [bold]{RESULTS_PATH}[/bold][/dim]\n"
    )


def _print_plain_table(results: dict, reports: list[InvestigationReport]) -> None:
    """Fallback plain-text table when Rich is not available."""
    m = results["metrics"]
    rows = [
        ("Causal Attribution Accuracy", "≥ 80%",
         f"{m['causal_attribution_accuracy'].get('accuracy', 0):.1%}",
         m['causal_attribution_accuracy'].get("pass")),
        ("Evidence Recall", "≥ 85%",
         f"{m['evidence_recall'].get('average_recall', 0):.1%}",
         m['evidence_recall'].get("pass")),
        ("Faithfulness Score", "≥ 90%",
         f"{m['faithfulness_score'].get('average_score', 0):.1%}",
         m['faithfulness_score'].get("pass")),
        ("Time to Explanation", "≤ 300 s",
         f"{m['time_to_explanation'].get('average_seconds', 0):.1f}s avg",
         m['time_to_explanation'].get("pass")),
        ("Human Audit Pass Rate", "≥ 70%",
         f"{m['human_audit_pass_rate'].get('pass_rate', 0):.1%}",
         m['human_audit_pass_rate'].get("pass")),
        ("Time to Insight",       "< 3 min", "manual", None),
        ("Feedback Loop Quality", "manual",  "descriptive", None),
    ]
    hdr = f"{'Metric':<35} {'Target':<18} {'Actual':<18} {'Pass'}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for name, target, actual, passed in rows:
        pf = "PASS" if passed else ("FAIL" if passed is False else "N/A")
        print(f"{name:<35} {target:<18} {actual:<18} {pf}")
    print("=" * len(hdr))
    print(f"\nEstimated API cost: ${_estimate_cost(reports):.4f} USD\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CostSherlock evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the pipeline run and load cached reports from reports/eval/",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def _load_cached_reports() -> list[InvestigationReport]:
    """Load previously-generated reports from disk for re-evaluation."""
    from agents.sentinel import Sentinel
    from agents.detective import Detective
    from pipeline import CostSherlockPipeline, _extract_remediation

    logger.info("Loading cached reports from %s…", REPORTS_DIR)
    report_dir = Path(REPORTS_DIR)
    if not report_dir.exists() or not list(report_dir.glob("*.md")):
        raise FileNotFoundError(
            f"No cached reports found in {REPORTS_DIR}. "
            "Run without --skip-pipeline first."
        )

    # Re-run the pipeline in evaluation mode to get InvestigationReport objects.
    # We need the structured data, not just the markdown files.
    logger.info(
        "Note: --skip-pipeline still runs LLM agents to get structured data. "
        "Use it only to avoid re-detecting anomalies."
    )
    return run_pipeline()


def main() -> None:
    """Entry point: parse CLI args, run the pipeline, compute and display all 7 metrics."""
    args = _build_arg_parser().parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    t_start = time.monotonic()

    # ── Run or load pipeline ──────────────────────────────────────────────────
    if args.skip_pipeline:
        reports = _load_cached_reports()
    else:
        reports = run_pipeline()

    if not reports:
        logger.error("No reports generated — cannot evaluate. Exiting.")
        sys.exit(1)

    # ── Run metrics ───────────────────────────────────────────────────────────
    results = run_evaluation(reports)

    # ── Attach metadata ───────────────────────────────────────────────────────
    results["meta"] = {
        "reports_evaluated": len(reports),
        "estimated_api_cost_usd": round(_estimate_cost(reports), 6),
        "total_eval_seconds": round(time.monotonic() - t_start, 2),
        "ground_truth_path": GT_PATH,
        "feedback_dir": FEEDBACK_DIR,
    }

    # ── Print Rich table ──────────────────────────────────────────────────────
    _print_rich_table(results, reports)

    # ── Save results.json ─────────────────────────────────────────────────────
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved evaluation results → %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
