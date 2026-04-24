"""CostSherlock Demo — Rich terminal UI over synthetic data."""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from agents import InvestigationReport
from agents.analyst import Analyst
from agents.detective import Detective
from agents.narrator import Narrator
from agents.sentinel import Sentinel
from pipeline import CostSherlockPipeline, _safe_filename

# ── Logging: suppress noisy INFO from sub-libraries during demo ───────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(name)s  %(message)s",
)
# Keep CostSherlock agents at INFO so progress is visible in the log file.
for _name in ("agents.sentinel", "agents.detective", "agents.analyst", "agents.narrator", "pipeline"):
    logging.getLogger(_name).setLevel(logging.INFO)

# ── Paths ─────────────────────────────────────────────────────────────────────
COST_DATA   = "data/synthetic/demo_cost.json"
CLOUDTRAIL  = "data/synthetic/demo_cloudtrail"
OUTPUT_DIR  = Path("reports/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Approximate token costs for claude-sonnet-4-6 ─────────────────────
# $3.00 / 1M input tokens,  $15.00 / 1M output tokens
_INPUT_COST_PER_TOKEN  = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

# Rough character-to-token ratio for estimating without a tokenizer.
_CHARS_PER_TOKEN = 4.0

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "green"
    if conf >= 0.55:
        return "yellow"
    return "red"


def _confidence_bar(conf: float, width: int = 10) -> str:
    filled = round(conf * width)
    return "[green]" + "█" * filled + "[/green][dim]" + "░" * (width - filled) + "[/dim]"


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Run the full CostSherlock pipeline on bundled synthetic data and print a Rich summary."""
    demo_start = time.monotonic()

    # ── Header panel ─────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]CostSherlock[/bold cyan]  [dim]v0.1 — synthetic data demo[/dim]\n"
            "[dim]Multi-agent AWS cost anomaly investigation[/dim]\n\n"
            f"[dim]Cost data :[/dim] [white]{COST_DATA}[/white]\n"
            f"[dim]CloudTrail:[/dim] [white]{CLOUDTRAIL}[/white]\n"
            f"[dim]Output    :[/dim] [white]{OUTPUT_DIR}[/white]",
            title="[bold white] Investigation Started [/bold white]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    # ── Stage 1: Sentinel ─────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Stage 1/4 — Sentinel: detecting anomalies…", total=None)
        df = Sentinel.load_from_json(COST_DATA)
        anomalies = Sentinel.detect_anomalies(df)
        anomalies = anomalies[:5]   # top-5 by z-score
        progress.update(task, description=f"[green]Stage 1/4 — Sentinel: {len(anomalies)} anomalies detected")
        time.sleep(0.3)  # let the completion message render

    console.print(
        f"  [green]✓[/green] Sentinel detected [bold]{len(anomalies)}[/bold] anomaly/anomalies "
        f"across [bold]{df['service'].nunique()}[/bold] services "
        f"([bold]{len(df)}[/bold] cost records loaded)"
    )

    # ── Stage 2: Detective ────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Stage 2/4 — Detective: loading CloudTrail logs…", total=None)
        ct_events = Detective.load_cloudtrail_logs(CLOUDTRAIL)
        progress.update(task, description=f"[green]Stage 2/4 — Detective: {len(ct_events)} events loaded")
        time.sleep(0.3)

    console.print(
        f"  [green]✓[/green] Detective loaded [bold]{len(ct_events)}[/bold] CloudTrail events"
    )

    # ── Stages 3 & 4: Analyst + Narrator (per anomaly) ───────────────────────
    console.print()
    analyst  = Analyst()
    narrator = Narrator()

    reports:           list[InvestigationReport] = []
    total_input_chars  = 0
    total_output_chars = 0

    for idx, anomaly in enumerate(anomalies, start=1):
        console.print(
            Panel(
                f"[bold]{anomaly.service}[/bold]   "
                f"[dim]{anomaly.date}[/dim]   "
                f"delta=[bold red]${anomaly.delta:.2f}[/bold red]   "
                f"z=[yellow]{anomaly.z_score:.2f}[/yellow]",
                title=f"[white] Anomaly {idx}/{len(anomalies)} [/white]",
                border_style="blue",
                padding=(0, 2),
            )
        )
        anomaly_start = time.monotonic()

        # Detective — correlate
        suspects = Detective.get_events_in_window(ct_events, anomaly)
        console.print(
            f"    [dim]Detective:[/dim] [bold]{len(suspects)}[/bold] suspect event(s) in window"
        )

        # Analyst
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "    [cyan]Stage 3/4 — Analyst: RAG retrieval + LLM reasoning…",
                total=None,
            )
            analysis = analyst.analyze(anomaly, suspects)
            progress.update(task, description="    [green]Stage 3/4 — Analyst: complete")
            time.sleep(0.2)

        hypotheses = analysis.get("hypotheses", [])
        ruled_out  = analysis.get("ruled_out", [])
        top        = hypotheses[0] if hypotheses else None

        console.print(
            f"    [dim]Analyst:[/dim] [bold]{len(hypotheses)}[/bold] hypothesis/hypotheses, "
            f"[bold]{len(ruled_out)}[/bold] ruled out"
        )
        if top:
            conf_color = _confidence_color(top.confidence)
            console.print(
                f"    [dim]Top hypothesis:[/dim] [{conf_color}]{top.root_cause[:90]}[/{conf_color}]"
            )

        # Narrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "    [cyan]Stage 4/4 — Narrator: generating cited report…",
                total=None,
            )
            report_md = narrator.generate_report(anomaly, analysis)
            progress.update(task, description="    [green]Stage 4/4 — Narrator: complete")
            time.sleep(0.2)

        # Track token estimates
        total_input_chars  += len(str(analysis))
        total_output_chars += len(report_md)

        # Save to disk
        filename   = _safe_filename(anomaly.service, anomaly.date) + ".md"
        save_path  = OUTPUT_DIR / filename
        save_path.write_text(report_md, encoding="utf-8")

        elapsed = round(time.monotonic() - anomaly_start, 1)
        console.print(
            f"    [dim]Narrator:[/dim] report saved → "
            f"[underline]{save_path}[/underline]  [dim]({elapsed}s)[/dim]"
        )

        # Assemble InvestigationReport
        from pipeline import _extract_remediation
        inv = InvestigationReport(
            anomaly=anomaly,
            hypotheses=hypotheses,
            ruled_out=ruled_out,
            remediation=_extract_remediation(report_md),
            overall_confidence=top.confidence if top else 0.0,
            report_markdown=report_md,
            elapsed_seconds=elapsed,
        )
        reports.append(inv)
        console.print()

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(
        title="[bold]Investigation Summary[/bold]",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("Service",     style="bold white",  no_wrap=True)
    table.add_column("Date",        style="dim",         no_wrap=True)
    table.add_column("Delta",       style="red",         justify="right", no_wrap=True)
    table.add_column("Root Cause",  style="white",       ratio=4)
    table.add_column("Confidence",  justify="center",    no_wrap=True)
    table.add_column("Time",        style="dim",         justify="right", no_wrap=True)

    for r in reports:
        top    = r.hypotheses[0] if r.hypotheses else None
        conf   = top.confidence if top else 0.0
        cause  = (top.root_cause[:75] + "…") if top and len(top.root_cause) > 75 else (top.root_cause if top else "INSUFFICIENT_EVIDENCE")
        color  = _confidence_color(conf)
        bar    = _confidence_bar(conf)
        table.add_row(
            r.anomaly.service,
            r.anomaly.date,
            f"${r.anomaly.delta:.2f}",
            cause,
            f"{bar} [{color}]{conf:.0%}[/{color}]",
            f"{r.elapsed_seconds:.1f}s",
        )

    console.print(table)
    console.print()

    # ── Cost estimate ─────────────────────────────────────────────────────────
    input_tokens  = _estimate_tokens("x" * total_input_chars)
    output_tokens = _estimate_tokens("x" * total_output_chars)
    # Each anomaly = 2 LLM calls (Analyst + Narrator); input is roughly 3× output
    total_input_tokens  = input_tokens * 3
    total_output_tokens = output_tokens
    est_cost = (
        total_input_tokens  * _INPUT_COST_PER_TOKEN +
        total_output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    console.print(
        Panel(
            f"[bold]API calls      :[/bold]  {len(reports) * 2}  "
            f"([dim]2 per anomaly: Analyst + Narrator[/dim])\n"
            f"[bold]Est. input tok :[/bold]  ~{total_input_tokens:,}\n"
            f"[bold]Est. output tok:[/bold]  ~{total_output_tokens:,}\n"
            f"[bold]Est. API cost  :[/bold]  [green]~${est_cost:.4f}[/green]  "
            f"[dim](claude-sonnet-4: $3/1M in · $15/1M out)[/dim]\n"
            f"[bold]Total elapsed  :[/bold]  {time.monotonic() - demo_start:.1f}s\n"
            f"[bold]Reports saved  :[/bold]  {OUTPUT_DIR}/",
            title="[white] Run Statistics [/white]",
            border_style="green",
            padding=(0, 2),
        )
    )

    # ── Render first report ───────────────────────────────────────────────────
    if reports:
        console.print()
        console.rule("[bold cyan]Full Report — Highest-Confidence Anomaly[/bold cyan]")
        console.print()
        best = max(reports, key=lambda r: r.overall_confidence)
        console.print(Markdown(best.report_markdown))
        console.print()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print_exception()
        logging.getLogger(__name__).error("Demo failed: %s", exc)
        sys.exit(1)
