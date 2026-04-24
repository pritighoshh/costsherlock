"""CostSherlock Streamlit Dashboard — main application."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env with override=True so the project key always wins over any stale
# value already present in the OS / shell environment.
load_dotenv(dotenv_path=_ROOT / ".env", override=True)

from agents import Anomaly, InvestigationReport, RuledOutEvent  # noqa: E402
from agents.analyst import Analyst  # noqa: E402
from agents.detective import Detective  # noqa: E402
from agents.narrator import Narrator  # noqa: E402
from agents.sentinel import Sentinel  # noqa: E402
from pipeline import _extract_remediation  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DEMO_COST_PATH = str(_ROOT / "data" / "synthetic" / "demo_cost.json")
DEMO_CLOUDTRAIL_DIR = str(_ROOT / "data" / "synthetic" / "demo_cloudtrail")
MODEL_NAME = "claude-sonnet-4-6"
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000
_CHARS_PER_TOKEN = 4.0

NAVY = "#1B2A4A"
BLUE = "#2563EB"
GREEN = "#059669"
RED = "#DC2626"
ORANGE = "#F97316"
AMBER = "#D97706"
VIEWS = ["Timeline", "Investigation", "Evidence", "Compare", "Feedback"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CostSherlock",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* ── Header bar ───────────────────────────────────────────────────── */
        [data-testid="stHeader"] {{
            background-color: {NAVY};
            border-bottom: 2px solid #2563EB;
        }}
        /* Keep the Streamlit hamburger/deploy icons visible */
        [data-testid="stHeader"] button svg {{
            fill: #CBD5E1 !important;
        }}

        /* ── Main content area ────────────────────────────────────────────── */
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}

        /* ── Sidebar gradient background ─────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F1E38 0%, #1B2A4A 60%, #1e3a5f 100%);
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: #CBD5E1 !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
        }}
        /* Brighten radio labels on hover */
        [data-testid="stSidebar"] [role="radio"]:hover label {{
            color: #FFFFFF !important;
        }}

        /* ── Sidebar brand block ─────────────────────────────────────────── */
        .sb-brand {{
            background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(5,150,105,0.15));
            border: 1px solid rgba(96,165,250,0.3);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
        }}
        .sb-brand-title {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #FFFFFF !important;
            letter-spacing: -0.01em;
        }}
        .sb-brand-sub {{
            font-size: 0.72rem;
            color: #93C5FD !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        /* ── Sidebar stat pill ───────────────────────────────────────────── */
        .sb-stat {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 7px 12px;
            margin-bottom: 6px;
        }}
        .sb-stat-label {{
            font-size: 0.75rem;
            color: #94A3B8 !important;
        }}
        .sb-stat-value {{
            font-size: 0.85rem;
            font-weight: 700;
            color: #E2E8F0 !important;
        }}

        /* ── Status banner in Investigation view ──────────────────────────── */
        .status-banner {{
            border-radius: 8px;
            padding: 10px 18px;
            margin-bottom: 16px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.03em;
        }}
        .status-high {{
            background: rgba(5,150,105,0.12);
            border: 1px solid #059669;
            color: #059669;
        }}
        .status-medium {{
            background: rgba(217,119,6,0.10);
            border: 1px solid #D97706;
            color: #D97706;
        }}
        .status-low {{
            background: rgba(220,38,38,0.10);
            border: 1px solid #DC2626;
            color: #DC2626;
        }}

        /* ── Severity badge ──────────────────────────────────────────────── */
        .sev-badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #FFFFFF;
        }}
        .sev-critical {{ background: #DC2626; }}
        .sev-warning {{ background: #D97706; }}
        .sev-info {{ background: #2563EB; }}

        /* ── Green "View Report" button wrapper ──────────────────────────── */
        div.btn-investigated button {{
            background-color: #059669 !important;
            border-color: #059669 !important;
            color: white !important;
        }}
        div.btn-investigated button:hover {{
            background-color: #047857 !important;
        }}

        /* ── Shimmer loading animation ───────────────────────────────────── */
        @keyframes shimmer {{
            0% {{ background-position: -1000px 0; }}
            100% {{ background-position: 1000px 0; }}
        }}
        .loading-shimmer {{
            background: linear-gradient(90deg, var(--secondary-background-color) 25%, rgba(128,128,128,0.1) 50%, var(--secondary-background-color) 75%);
            background-size: 1000px 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 8px;
            height: 16px;
            margin-bottom: 8px;
        }}

        /* ── Anomaly callout card ─────────────────────────────────────────── */
        .anomaly-callout {{
            background: var(--secondary-background-color);
            color: var(--text-color);
            border-left: 5px solid {RED};
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(220, 38, 38, 0.12);
            padding: 16px 22px;
            margin-bottom: 18px;
        }}

        /* ── Generic info card (used in Compare, Evidence, Feedback) ─────── */
        .info-card {{
            background: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 10px;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.1);
            padding: 18px 22px;
            margin-bottom: 14px;
            border-top: 3px solid {BLUE};
        }}

        /* ── Anomaly table ────────────────────────────────────────────────── */
        .tbl-row {{
            border-bottom: 1px solid rgba(128, 128, 128, 0.15);
            padding: 4px 0;
            transition: background 0.1s;
        }}
        .tbl-row-high {{
            background: rgba(220, 38, 38, 0.06);
            border-bottom: 1px solid rgba(220, 38, 38, 0.2);
        }}

        /* ── Cost calculation monospace box ──────────────────────────────── */
        .cost-box {{
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid {GREEN};
            border-radius: 8px;
            padding: 12px 16px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.87rem;
            white-space: pre-wrap;
            margin: 8px 0 12px 0;
        }}

        /* ── Category badge pill ─────────────────────────────────────────── */
        .cat-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.73rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }}

        /* ── Metric cards ────────────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: rgba(37, 99, 235, 0.07) !important;
            border-radius: 10px;
            box-shadow: 0 1px 5px rgba(0, 0, 0, 0.07);
            padding: 14px 16px !important;
            border-left: 3px solid {BLUE};
        }}
        [data-testid="stMetric"] * {{
            color: var(--text-color) !important;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] *,
        [data-testid="stMetric"] div[class*="metric-container"] div:nth-child(2) * {{
            font-weight: 700 !important;
        }}

        /* ── Download button ─────────────────────────────────────────────── */
        [data-testid="stDownloadButton"] button {{
            border-color: {BLUE} !important;
            color: {BLUE} !important;
        }}

        /* ── Footer ──────────────────────────────────────────────────────── */
        .cs-footer {{
            text-align: center;
            color: var(--text-color);
            opacity: 0.55;
            font-size: 0.78rem;
            padding: 24px 0 8px 0;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            margin-top: 48px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "data_loaded": False,
    "cost_df": None,
    "anomalies": [],
    "selected_anomaly": None,
    "investigations": [],
    "current_investigation": None,
    "cloudtrail_logs": [],
    "api_calls": 0,
    "total_cost_estimate": 0.0,
    "current_view": "Timeline",
    "z_threshold_slider": 2.5,
    # deferred toast: set before st.rerun(), consumed once at next render
    "_pending_toast": "",
    # auto-run flag: set when "Investigate" clicked from Timeline
    "_auto_run": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Consume any pending toast immediately (must be before other UI) ────────────
if st.session_state._pending_toast:
    st.toast(st.session_state._pending_toast, icon="✅")
    st.session_state._pending_toast = ""

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _get_analyst() -> Analyst:
    return Analyst()


@st.cache_resource
def _get_narrator() -> Narrator:
    return Narrator()


@st.cache_data
def _load_cost_data(path: str) -> pd.DataFrame:
    return Sentinel.load_from_json(path)


@st.cache_data
def _load_cloudtrail(log_dir: str) -> list[dict]:
    return Detective.load_cloudtrail_logs(log_dir)


@st.cache_data
def _detect_anomalies_cached(path: str, z_threshold: float) -> list[dict]:
    df = _load_cost_data(path)
    return [a.model_dump() for a in Sentinel.detect_anomalies(df, z_threshold=z_threshold)]


# ── Helper utilities ──────────────────────────────────────────────────────────

def _estimate_api_cost(input_text: str, output_text: str) -> float:
    in_tok = len(input_text) / _CHARS_PER_TOKEN
    out_tok = len(output_text) / _CHARS_PER_TOKEN
    return in_tok * _INPUT_COST_PER_TOKEN + out_tok * _OUTPUT_COST_PER_TOKEN


_BADGE_COLORS: dict[str, str] = {
    "WRONG_MECHANISM": RED,
    "TEMPORAL_ONLY": ORANGE,
    "INSUFFICIENT_EVIDENCE": "#6B7280",
    "UNRELATED_SERVICE": "#8B5CF6",
}


def _category_badge(category: str) -> str:
    color = _BADGE_COLORS.get(category, "#6B7280")
    return (
        f'<span class="cat-badge" style="background:{color}">'
        f"{category}</span>"
    )


def _ensure_cloudtrail() -> list[dict]:
    """Return CloudTrail events from session state, loading demo data as fallback."""
    if st.session_state.cloudtrail_logs:
        return st.session_state.cloudtrail_logs
    ct = _load_cloudtrail(DEMO_CLOUDTRAIL_DIR)
    st.session_state.cloudtrail_logs = ct
    return ct


def _persist_investigation(inv: InvestigationReport) -> None:
    """Store an InvestigationReport in session state, deduplicating by (service, date)."""
    st.session_state.current_investigation = inv
    existing = {(r.anomaly.service, r.anomaly.date) for r in st.session_state.investigations}
    if (inv.anomaly.service, inv.anomaly.date) not in existing:
        st.session_state.investigations.append(inv)


def _severity_badge(delta: float) -> str:
    """Return an HTML severity badge based on cost delta."""
    if delta > 200:
        return '<span class="sev-badge sev-critical">Critical</span>'
    elif delta > 50:
        return '<span class="sev-badge sev-warning">Warning</span>'
    return '<span class="sev-badge sev-info">Info</span>'


def _confidence_color(conf: float) -> str:
    """Return a semantic color string for a confidence value."""
    if conf >= 0.8:
        return GREEN
    if conf >= 0.55:
        return AMBER
    return RED


def _conf_class(conf: float) -> str:
    """Return the CSS class name for a confidence-level status banner."""
    if conf >= 0.8:
        return "status-high"
    if conf >= 0.55:
        return "status-medium"
    return "status-low"


def _conf_label(conf: float) -> str:
    """Return a human-readable confidence label with percentage."""
    if conf >= 0.8:
        return f"HIGH CONFIDENCE ({conf:.0%})"
    if conf >= 0.55:
        return f"MEDIUM CONFIDENCE ({conf:.0%})"
    return f"LOW CONFIDENCE ({conf:.0%})"


# ── Core pipeline logic (no Streamlit UI side-effects) ───────────────────────

def _pipeline_core(
    anomaly: Anomaly,
    ct_events: list[dict],
) -> tuple[InvestigationReport, str] | tuple[None, str]:
    """Run Sentinel→Detective→Analyst→Narrator for one anomaly.

    Returns:
        (InvestigationReport, "") on success, or (None, error_message) on failure.
    """
    t_start = time.monotonic()
    try:
        suspects = Detective.get_events_in_window(ct_events, anomaly)

        analysis = _get_analyst().analyze(anomaly, suspects)
        st.session_state.api_calls += 1

        hypotheses = analysis.get("hypotheses", [])
        ruled_out = analysis.get("ruled_out", [])

        report_md = _get_narrator().generate_report(anomaly, analysis)
        st.session_state.api_calls += 1

    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline error for %s", anomaly.service)
        return None, str(exc)

    elapsed = round(time.monotonic() - t_start, 2)
    cost_estimate = _estimate_api_cost(str(analysis) * 3, report_md)
    st.session_state.total_cost_estimate += cost_estimate

    inv = InvestigationReport(
        anomaly=anomaly,
        hypotheses=hypotheses,
        ruled_out=ruled_out,
        remediation=_extract_remediation(report_md),
        overall_confidence=hypotheses[0].confidence if hypotheses else 0.0,
        report_markdown=report_md,
        elapsed_seconds=elapsed,
    )
    return inv, ""


# ── Interactive single-anomaly runner (shows st.status with steps) ────────────

def _run_investigation(anomaly: Anomaly) -> None:
    """Execute the pipeline with live st.status progress, then rerun."""
    try:
        ct_events = _ensure_cloudtrail()
    except Exception as exc:
        st.error(f"Could not load CloudTrail logs: {exc}")
        return

    t_start = time.monotonic()

    try:
        with st.status("Running investigation pipeline…", expanded=True) as status:

            # Step 1 — Sentinel (instant)
            st.write("🔍 **Sentinel:** Anomaly confirmed")
            st.write(
                f"   Service `{anomaly.service}` · Date `{anomaly.date}` · "
                f"z = {anomaly.z_score:.2f} · δ = +${anomaly.delta:.2f}"
            )

            # Step 2 — Detective (fast, no LLM)
            st.write("🕵️ **Detective:** Correlating CloudTrail events…")
            suspects = Detective.get_events_in_window(ct_events, anomaly)
            st.write(f"   Found **{len(suspects)}** suspect event(s) in ±48 h window")

            # Step 3 — Analyst (slow LLM call)
            st.write("🧠 **Analyst:** Querying RAG knowledge base and reasoning…")
            with st.spinner("Analyst thinking — this may take 20–40 s…"):
                analysis = _get_analyst().analyze(anomaly, suspects)
            st.session_state.api_calls += 1
            hypotheses = analysis.get("hypotheses", [])
            ruled_out = analysis.get("ruled_out", [])
            st.write(
                f"   Generated **{len(hypotheses)}** hypothesis/hypotheses · "
                f"**{len(ruled_out)}** ruled out"
            )

            # Step 4 — Narrator (slow LLM call)
            st.write("📝 **Narrator:** Drafting cited investigation report…")
            with st.spinner("Narrator writing — usually 15–30 s…"):
                report_md = _get_narrator().generate_report(anomaly, analysis)
            st.session_state.api_calls += 1

            status.update(label="✅ Investigation complete!", state="complete")

    except Exception as exc:
        st.error(f"Investigation failed: {exc}")
        logging.getLogger(__name__).exception("Investigation error")
        return

    elapsed = round(time.monotonic() - t_start, 2)
    cost_estimate = _estimate_api_cost(str(analysis) * 3, report_md)
    st.session_state.total_cost_estimate += cost_estimate

    inv = InvestigationReport(
        anomaly=anomaly,
        hypotheses=hypotheses,
        ruled_out=ruled_out,
        remediation=_extract_remediation(report_md),
        overall_confidence=hypotheses[0].confidence if hypotheses else 0.0,
        report_markdown=report_md,
        elapsed_seconds=elapsed,
    )
    _persist_investigation(inv)

    top_conf = f"{hypotheses[0].confidence:.0%}" if hypotheses else "n/a"
    st.session_state._pending_toast = (
        f"Investigation done · {anomaly.service} · {anomaly.date} · "
        f"confidence {top_conf} · {elapsed:.1f}s"
    )
    st.rerun()


# ── Batch runner (used by "Run All Investigations") ───────────────────────────

def _run_all_investigations(anomalies: list[Anomaly]) -> None:
    """Process every anomaly sequentially with a progress bar, then rerun."""
    try:
        ct_events = _ensure_cloudtrail()
    except Exception as exc:
        st.error(f"Could not load CloudTrail logs: {exc}")
        return

    total = len(anomalies)
    if total == 0:
        st.warning("No anomalies to investigate.")
        return

    completed = 0
    failed: list[str] = []

    progress = st.progress(0.0, text="Starting batch investigation…")
    status_text = st.empty()

    for i, anomaly in enumerate(anomalies):
        progress.progress(i / total, text=f"Investigating {anomaly.service} ({i + 1}/{total})…")
        status_text.markdown(
            f"⏳ **{anomaly.service}** · {anomaly.date} · "
            f"z = {anomaly.z_score:.2f}"
        )

        # Skip if already investigated in this session
        existing_keys = {(r.anomaly.service, r.anomaly.date) for r in st.session_state.investigations}
        if (anomaly.service, anomaly.date) in existing_keys:
            status_text.markdown(f"⏭️ **{anomaly.service}** — already investigated, skipping")
            completed += 1
            continue

        with st.spinner(f"Running pipeline for {anomaly.service}…"):
            inv, err = _pipeline_core(anomaly, ct_events)

        if inv is not None:
            _persist_investigation(inv)
            completed += 1
        else:
            failed.append(f"{anomaly.service} ({anomaly.date}): {err}")

    progress.progress(1.0, text="Batch complete!")
    status_text.empty()

    if failed:
        st.warning(
            f"**{completed}/{total} succeeded.** Failed:\n"
            + "\n".join(f"- {f}" for f in failed)
        )
        st.session_state._pending_toast = (
            f"Batch done: {completed}/{total} succeeded, {len(failed)} failed"
        )
    else:
        st.session_state._pending_toast = (
            f"All {completed} investigations complete!"
        )

    st.rerun()


# ── Investigation navigation callbacks ───────────────────────────────────────
# Defined as module-level functions so Streamlit's on_click/on_change machinery
# commits state BEFORE the automatic rerun — avoiding the race condition where
# the sidebar radio reads stale state and resets current_view to "Timeline".

def _nav_prev() -> None:
    invs_list = st.session_state.investigations
    inv = st.session_state.current_investigation
    if inv is None or not invs_list:
        return
    current_idx = next(
        (i for i, r in enumerate(invs_list)
         if r.anomaly.service == inv.anomaly.service and r.anomaly.date == inv.anomaly.date),
        0,
    )
    if current_idx > 0:
        target = invs_list[current_idx - 1]
        st.session_state.current_investigation = target
        st.session_state.selected_anomaly = target.anomaly
        # Keep selectbox widget state in sync so it doesn't lag behind
        inv_labels = [f"{r.anomaly.service}  —  {r.anomaly.date}" for r in invs_list]
        st.session_state.inv_nav_sel = inv_labels[current_idx - 1]


def _nav_next() -> None:
    invs_list = st.session_state.investigations
    inv = st.session_state.current_investigation
    if inv is None or not invs_list:
        return
    current_idx = next(
        (i for i, r in enumerate(invs_list)
         if r.anomaly.service == inv.anomaly.service and r.anomaly.date == inv.anomaly.date),
        0,
    )
    if current_idx < len(invs_list) - 1:
        target = invs_list[current_idx + 1]
        st.session_state.current_investigation = target
        st.session_state.selected_anomaly = target.anomaly
        inv_labels = [f"{r.anomaly.service}  —  {r.anomaly.date}" for r in invs_list]
        st.session_state.inv_nav_sel = inv_labels[current_idx + 1]


def _on_investigate(anomaly: Anomaly) -> None:
    """Callback for both 'Investigate' and 'View Report' buttons on the Timeline.

    Sets all required state BEFORE the rerun so the sidebar radio (key='current_view')
    hasn't been instantiated yet when current_view is written — avoiding the
    StreamlitAPIException that firing a direct assignment after widget render causes.
    """
    st.session_state.selected_anomaly = anomaly
    st.session_state.current_view = "Investigation"
    cached = next(
        (r for r in st.session_state.investigations
         if r.anomaly.service == anomaly.service and r.anomaly.date == anomaly.date),
        None,
    )
    if cached:
        st.session_state.current_investigation = cached
        st.session_state._auto_run = False
    else:
        st.session_state._auto_run = True


def _nav_select() -> None:
    invs_list = st.session_state.investigations
    sel_label = st.session_state.get("inv_nav_sel", "")
    if not sel_label or not invs_list:
        return
    inv_labels = [f"{r.anomaly.service}  —  {r.anomaly.date}" for r in invs_list]
    if sel_label not in inv_labels:
        return
    target = invs_list[inv_labels.index(sel_label)]
    st.session_state.current_investigation = target
    st.session_state.selected_anomaly = target.anomaly


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div class="sb-brand">
    <div class="sb-brand-title">🔍 CostSherlock</div>
    <div class="sb-brand-sub">AWS FinOps · Anomaly Intelligence</div>
</div>
""", unsafe_allow_html=True)

    # key="current_view" makes st.session_state.current_view the single source of
    # truth for the radio.  Programmatic writes (e.g. setting current_view =
    # "Investigation" from the Timeline) are reflected immediately on the next
    # render without the index= / widget-state conflict that caused nav resets.
    st.radio("Navigation", VIEWS, key="current_view")

    st.divider()

    # ── Investigation status pills ────────────────────────────────────────────
    anomalies_ss = st.session_state.anomalies
    inv_count = len(st.session_state.investigations)
    total_count = len(anomalies_ss)
    total_delta = sum(a.delta for a in anomalies_ss)
    status_icon = "✅" if (total_count > 0 and inv_count == total_count) else "⏳"
    if total_count > 0:
        st.markdown(f"""
    <div class="sb-stat">
        <span class="sb-stat-label">Investigations</span>
        <span class="sb-stat-value">{status_icon} {inv_count}/{total_count}</span>
    </div>
    <div class="sb-stat">
        <span class="sb-stat-label">Total Cost Impact</span>
        <span class="sb-stat-value" style="color:#FCA5A5 !important">+${total_delta:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)
        st.markdown("")

    # ── Load Demo Data ────────────────────────────────────────────────────────
    if st.button("📥 Load Demo Data", use_container_width=True):
        with st.spinner("Loading synthetic dataset…"):
            try:
                z = st.session_state.z_threshold_slider
                df = _load_cost_data(DEMO_COST_PATH)
                ct = _load_cloudtrail(DEMO_CLOUDTRAIL_DIR)
                anomaly_dicts = _detect_anomalies_cached(DEMO_COST_PATH, z)
                anomalies_loaded = [Anomaly(**a) for a in anomaly_dicts]
                st.session_state.update(
                    {
                        "data_loaded": True,
                        "cost_df": df,
                        "cloudtrail_logs": ct,
                        "anomalies": anomalies_loaded,
                    }
                )
                st.success(
                    f"✓ {len(anomalies_loaded)} anomaly/anomalies across "
                    f"{df['service'].nunique()} services"
                )
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    # ── Custom file upload ────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload Cost JSON", type=["json"])
    if uploaded is not None:
        with st.spinner("Parsing uploaded file…"):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False, mode="wb"
                ) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                df = Sentinel.load_from_json(tmp_path)
                z = st.session_state.z_threshold_slider
                anomalies_up = Sentinel.detect_anomalies(df, z_threshold=z)
                st.session_state.update(
                    {
                        "data_loaded": True,
                        "cost_df": df,
                        "anomalies": anomalies_up,
                    }
                )
                st.success(f"✓ {len(anomalies_up)} anomaly/anomalies detected")
            except json.JSONDecodeError:
                st.error("Invalid JSON — check file format.")
            except ValueError as exc:
                st.error(f"Schema error: {exc}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        st.slider(
            "Z-Score Threshold",
            min_value=1.5,
            max_value=4.0,
            step=0.1,
            key="z_threshold_slider",
            help="Anomalies are flagged when cost exceeds the rolling mean by this many σ.",
        )

    st.divider()

    # ── Session stats ─────────────────────────────────────────────────────────
    st.caption(f"**Model:** `{MODEL_NAME}`")
    st.caption(f"**API calls:** {st.session_state.api_calls}")
    st.caption(f"**Est. API cost:** `${st.session_state.total_cost_estimate:.4f}`")
    st.caption(f"**Investigations:** {inv_count}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — dispatch on current_view
# ═════════════════════════════════════════════════════════════════════════════

view = st.session_state.current_view

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — ANOMALY TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
if view == "Timeline":
    st.title("📈 Anomaly Timeline")

    if not st.session_state.data_loaded:
        st.info(
            "👈 Click **Load Demo Data** in the sidebar to get started, "
            "or upload your own Cost Explorer JSON export."
        )
    else:
        df: pd.DataFrame = st.session_state.cost_df
        anomalies: list[Anomaly] = st.session_state.anomalies

        if df is None or df.empty:
            st.error("Loaded dataset is empty. Please reload or upload a different file.")
        else:
            # ── Summary metric cards ──────────────────────────────────────────
            total_delta = sum(a.delta for a in anomalies)
            avg_z = sum(a.z_score for a in anomalies) / len(anomalies) if anomalies else 0
            inv_done = len(st.session_state.investigations)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Anomalies", len(anomalies))
            m2.metric("Avg Z-Score", f"{avg_z:.2f}")
            m3.metric("Total Cost Impact", f"${total_delta:,.2f}")
            m4.metric("Investigated", f"{inv_done}/{len(anomalies)}")

            st.markdown("")

            # ── Plotly chart ──────────────────────────────────────────────────
            palette = px.colors.qualitative.Set2
            services = df["service"].unique().tolist()
            color_map = {s: palette[i % len(palette)] for i, s in enumerate(services)}

            fig = go.Figure()
            for service in services:
                svc_df = df[df["service"] == service].sort_values("date")
                fig.add_trace(
                    go.Scatter(
                        x=svc_df["date"],
                        y=svc_df["cost"],
                        mode="lines",
                        name=service,
                        line=dict(color=color_map[service], width=1.8),
                        hovertemplate=(
                            f"<b>{service}</b><br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Cost: $%{y:.2f}<extra></extra>"
                        ),
                    )
                )

            if anomalies:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.Timestamp(a.date) for a in anomalies],
                        y=[a.cost for a in anomalies],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            symbol="diamond",
                            size=14,
                            color=RED,
                            line=dict(width=2, color="darkred"),
                        ),
                        customdata=[
                            (
                                f"<b>⚠ ANOMALY</b><br>"
                                f"Service: {a.service}<br>"
                                f"Cost: ${a.cost:.2f}<br>"
                                f"Expected: ${a.expected_cost:.2f}<br>"
                                f"Z-Score: {a.z_score:.2f}<br>"
                                f"Delta: +${a.delta:.2f}"
                            )
                            for a in anomalies
                        ],
                        hovertemplate="%{customdata}<extra></extra>",
                    )
                )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                legend=dict(orientation="v", x=1.01, y=1.0, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="Date"),
                yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="Daily Cost ($)"),
                hovermode="closest",
                margin=dict(l=60, r=40, t=20, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Anomaly table ─────────────────────────────────────────────────
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.subheader(f"Detected Anomalies ({len(anomalies)})")
            with col_right:
                already_done_keys = {
                    (r.anomaly.service, r.anomaly.date)
                    for r in st.session_state.investigations
                }
                pending = [
                    a for a in anomalies
                    if (a.service, a.date) not in already_done_keys
                ]
                run_all_label = (
                    f"⚡ Run All ({len(pending)} pending)"
                    if pending
                    else "⚡ All Investigated"
                )
                run_all_disabled = len(pending) == 0
                if st.button(
                    run_all_label,
                    disabled=run_all_disabled,
                    type="primary",
                    use_container_width=True,
                    key="run_all_btn",
                ):
                    _run_all_investigations(anomalies)

            if not anomalies:
                st.info(
                    "No anomalies detected at the current z-score threshold "
                    f"({st.session_state.z_threshold_slider:.1f}). "
                    "Try lowering the threshold in ⚙️ Settings."
                )
            else:
                sorted_anomalies = sorted(anomalies, key=lambda a: a.z_score, reverse=True)

                # Styled table header — 8 columns now (added Status)
                hdr_cols = st.columns([2.5, 2, 1.3, 1.3, 1.1, 1.3, 1.2, 1.5])
                labels_hdr = ["Service", "Date", "Cost", "Expected", "Z-Score", "Delta ($)", "Status", "Action"]
                for col, lbl in zip(hdr_cols, labels_hdr):
                    col.markdown(
                        f"<span style='font-weight:700;color:var(--text-color);font-size:0.82rem;"
                        f"text-transform:uppercase;letter-spacing:0.04em'>{lbl}</span>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    "<hr style='margin:4px 0 6px 0;border:none;border-top:2px solid rgba(128,128,128,0.3)'>",
                    unsafe_allow_html=True,
                )

                for i, a in enumerate(sorted_anomalies):
                    is_high = a.z_score > 3.0
                    is_investigated = (a.service, a.date) in already_done_keys
                    row = st.columns([2.5, 2, 1.3, 1.3, 1.1, 1.3, 1.2, 1.5])
                    row[0].write(a.service)
                    row[1].write(a.date)
                    row[2].write(f"${a.cost:.2f}")
                    row[3].write(f"${a.expected_cost:.2f}")
                    if is_high:
                        row[4].markdown(f"**:red[{a.z_score:.2f}]**")
                    else:
                        row[4].write(f"{a.z_score:.2f}")
                    row[5].write(f"${a.delta:.2f}")
                    # Status column
                    if is_investigated:
                        row[6].markdown("✅ Done")
                    else:
                        row[6].markdown("⏳ Pending")
                    # Action button — on_click callback writes current_view before
                    # the radio widget is instantiated, satisfying Streamlit's rule
                    if is_investigated:
                        with row[7]:
                            st.markdown('<div class="btn-investigated">', unsafe_allow_html=True)
                            st.button(
                                "📄 View Report",
                                key=f"inv_{i}",
                                on_click=_on_investigate,
                                args=(a,),
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        row[7].button(
                            "🔍 Investigate",
                            key=f"inv_{i}",
                            on_click=_on_investigate,
                            args=(a,),
                        )
                    # Thin row separator
                    sep_color = "rgba(220,38,38,0.2)" if is_high else "rgba(128,128,128,0.12)"
                    st.markdown(
                        f"<hr style='margin:2px 0;border:none;border-top:1px solid {sep_color}'>",
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — LIVE INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Investigation":
    st.title("🕵️ Live Investigation")

    if not st.session_state.data_loaded:
        st.warning(
            "No data loaded. Go to the Timeline view and click **Load Demo Data** first."
        )
    elif st.session_state.selected_anomaly is None:
        st.warning("Select an anomaly from the **Timeline** view to begin an investigation.")
    else:
        anomaly: Anomaly = st.session_state.selected_anomaly

        # Rounded anomaly summary card
        st.markdown(
            f"""
            <div class="anomaly-callout">
                <div style="font-size:1.15rem;font-weight:700;margin-bottom:6px">
                    {anomaly.service}
                    &nbsp;<span style="font-weight:400;opacity:0.65;font-size:0.95rem">
                    {anomaly.date}</span>
                </div>
                <span style="margin-right:18px">
                    💵 Cost <strong>${anomaly.cost:.2f}</strong>
                    &nbsp;<span style="opacity:0.55">vs expected ${anomaly.expected_cost:.2f}</span>
                </span>
                <span style="margin-right:18px">
                    📊 Z-Score <strong style="color:{RED}">{anomaly.z_score:.2f}</strong>
                </span>
                <span>
                    📈 Delta <strong style="color:{RED}">+${anomaly.delta:.2f}</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        inv: InvestigationReport | None = st.session_state.current_investigation

        # Check full investigations list so batch-run reports are found
        if inv is None or inv.anomaly.service != anomaly.service or inv.anomaly.date != anomaly.date:
            cached = next(
                (r for r in st.session_state.investigations
                 if r.anomaly.service == anomaly.service and r.anomaly.date == anomaly.date),
                None,
            )
            if cached:
                st.session_state.current_investigation = cached
                inv = cached

        already_done = (
            inv is not None
            and inv.anomaly.service == anomaly.service
            and inv.anomaly.date == anomaly.date
        )

        if already_done:
            # ── Investigation navigation bar ──────────────────────────────────
            invs_list = st.session_state.investigations
            if len(invs_list) > 1:
                inv_labels = [f"{r.anomaly.service}  —  {r.anomaly.date}" for r in invs_list]
                current_idx = next(
                    (i for i, r in enumerate(invs_list)
                     if r.anomaly.service == inv.anomaly.service and r.anomaly.date == inv.anomaly.date),
                    0,
                )
                logging.getLogger(__name__).debug(
                    "Investigation nav: showing %s/%s  idx=%d  service=%s  date=%s",
                    current_idx + 1, len(invs_list), current_idx,
                    inv.anomaly.service, inv.anomaly.date,
                )
                nav_l, nav_c, nav_r = st.columns([1, 6, 1])
                with nav_l:
                    st.button(
                        "← Prev",
                        disabled=(current_idx == 0),
                        use_container_width=True,
                        key="inv_prev",
                        on_click=_nav_prev,   # state commits before rerun
                    )
                with nav_c:
                    st.selectbox(
                        "Select investigation",
                        inv_labels,
                        index=current_idx,
                        key="inv_nav_sel",
                        label_visibility="collapsed",
                        on_change=_nav_select,  # state commits before rerun
                    )
                with nav_r:
                    st.button(
                        "Next →",
                        disabled=(current_idx == len(invs_list) - 1),
                        use_container_width=True,
                        key="inv_next",
                        on_click=_nav_next,   # state commits before rerun
                    )
                st.markdown("")

            # ── Status banner + severity badge ────────────────────────────────
            sev_html = _severity_badge(inv.anomaly.delta)
            st.markdown(
                f"""
                <div class="status-banner {_conf_class(inv.overall_confidence)}">
                    ● {_conf_label(inv.overall_confidence)}
                    &nbsp;&nbsp;{sev_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Metrics strip ─────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Elapsed", f"{inv.elapsed_seconds:.1f}s")
            m2.metric("Confidence", f"{inv.overall_confidence:.0%}")
            m3.metric("Hypotheses", len(inv.hypotheses))
            m4.metric("Ruled Out", len(inv.ruled_out))

            st.markdown("")

            # ── Mini event timeline ───────────────────────────────────────────
            if inv.hypotheses:
                try:
                    anomaly_ts = pd.Timestamp(inv.anomaly.date)
                    tl_fig = go.Figure()
                    tl_fig.add_trace(go.Scatter(
                        x=[anomaly_ts],
                        y=[1],
                        mode="markers+text",
                        marker=dict(symbol="diamond", size=16, color=RED, line=dict(width=2, color="darkred")),
                        text=[f"⚠ {inv.anomaly.service}"],
                        textposition="top center",
                        name="Anomaly Spike",
                        hovertemplate=f"<b>Anomaly</b><br>{inv.anomaly.date}<br>Cost: ${inv.anomaly.cost:.2f}<extra></extra>",
                    ))
                    tl_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=120,
                        margin=dict(l=20, r=20, t=10, b=20),
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, title=""),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 2], title=""),
                    )
                    st.plotly_chart(tl_fig, use_container_width=True)
                except Exception:
                    pass  # Timeline is decorative; skip silently on any parse error

            st.divider()

            # ── Report tabs ───────────────────────────────────────────────────
            tab_sum, tab_ev, tab_ro, tab_rem = st.tabs(
                ["📋 Summary", "🔗 Evidence Chain", "🚫 Ruled Out", "🔧 Remediation"]
            )

            with tab_sum:
                if not inv.report_markdown:
                    st.error(
                        "The report is empty — the Narrator agent may have returned no content. "
                        "Check your API key and retry."
                    )
                else:
                    st.markdown(inv.report_markdown)

                dl_col, _ = st.columns([1, 3])
                with dl_col:
                    st.download_button(
                        label="⬇️ Download Report (.md)",
                        data=inv.report_markdown or "",
                        file_name=(
                            f"{inv.anomaly.date}_"
                            f"{inv.anomaly.service.replace(' ', '_').replace('/', '_')}.md"
                        ),
                        mime="text/markdown",
                    )

            with tab_ev:
                st.subheader(f"Hypotheses ({len(inv.hypotheses)})")
                if not inv.hypotheses:
                    st.warning(
                        "No hypotheses were generated for this anomaly. "
                        "This can happen when the LLM cannot establish causal evidence. "
                        "Check the raw report in the Summary tab for details."
                    )
                else:
                    for i_h in range(0, len(inv.hypotheses), 2):
                        ev_cols = st.columns(2)
                        for j_h, ev_col in enumerate(ev_cols):
                            idx_h = i_h + j_h
                            if idx_h >= len(inv.hypotheses):
                                break
                            h = inv.hypotheses[idx_h]
                            conf_color = _confidence_color(h.confidence)
                            ev_count = len(h.evidence) if h.evidence else 0
                            with ev_col:
                                with st.expander(
                                    f"#{h.rank}  {h.root_cause[:70]}{'…' if len(h.root_cause) > 70 else ''}",
                                    expanded=(h.rank == 1),
                                ):
                                    st.markdown(
                                        f'<div style="border-left:4px solid {conf_color};padding-left:10px">',
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(h.confidence, text=f"Confidence: {h.confidence:.0%}")
                                    st.caption(f"📎 {ev_count} evidence item(s)")
                                    if h.evidence:
                                        for item in h.evidence:
                                            st.markdown(f"- {item}")
                                    if h.cost_calculation:
                                        st.markdown("**Cost Calculation:**")
                                        st.markdown(
                                            f'<div class="cost-box">{h.cost_calculation}</div>',
                                            unsafe_allow_html=True,
                                        )
                                    if h.causal_mechanism:
                                        st.info(h.causal_mechanism)
                                    st.markdown('</div>', unsafe_allow_html=True)

            with tab_ro:
                st.subheader(f"Ruled Out Events ({len(inv.ruled_out)})")
                if not inv.ruled_out:
                    st.caption("No events were explicitly ruled out in this investigation.")
                else:
                    hdr_ro = st.columns([2, 1.5, 3])
                    hdr_ro[0].markdown("**Event**")
                    hdr_ro[1].markdown("**Category**")
                    hdr_ro[2].markdown("**Reason**")
                    st.markdown(
                        "<hr style='margin:4px 0 8px 0;border:none;border-top:1px solid rgba(128,128,128,0.2)'>",
                        unsafe_allow_html=True,
                    )
                    for ro in inv.ruled_out:
                        date_str = ro.event_time[:10] if ro.event_time else "N/A"
                        row_cols = st.columns([2, 1.5, 3])
                        row_cols[0].write(f"{ro.event_name}\n\n_{date_str}_")
                        row_cols[1].markdown(_category_badge(ro.category), unsafe_allow_html=True)
                        row_cols[2].write(ro.reason)
                        st.markdown(
                            "<hr style='margin:2px 0;border:none;border-top:1px solid rgba(128,128,128,0.1)'>",
                            unsafe_allow_html=True,
                        )

            with tab_rem:
                if inv.remediation:
                    st.markdown(inv.remediation)
                else:
                    st.info("No remediation steps extracted.")
                dl_rem_col, _ = st.columns([1, 3])
                with dl_rem_col:
                    st.download_button(
                        label="⬇️ Download Report (.md)",
                        data=inv.report_markdown or "",
                        file_name=(
                            f"{inv.anomaly.date}_"
                            f"{inv.anomaly.service.replace(' ', '_').replace('/', '_')}_full.md"
                        ),
                        mime="text/markdown",
                        key="dl_rem",
                    )

        elif st.session_state._auto_run:
            # Triggered from Timeline "Investigate" button — run immediately
            st.session_state._auto_run = False
            _run_investigation(anomaly)
        else:
            # User navigated here manually without selecting from Timeline
            if not st.session_state.cloudtrail_logs:
                st.info(
                    "ℹ️ No CloudTrail logs loaded — the demo CloudTrail dataset will be "
                    "used automatically when you run the investigation."
                )
            st.info(
                "Click **Run Investigation** to start the 4-agent pipeline. "
                "This makes two LLM API calls (Analyst + Narrator) and takes ~60–90 s."
            )
            if st.button("🚀 Run Investigation", type="primary"):
                _run_investigation(anomaly)

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — EVIDENCE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Evidence":
    st.title("🔬 Evidence Explorer")

    inv = st.session_state.current_investigation
    if not st.session_state.data_loaded:
        st.warning("Load data first using the sidebar.")
    elif inv is None:
        st.warning(
            "No investigation to display yet. "
            "Run one from the **Investigation** view."
        )
    else:
        # Investigation header card
        conf_color_ev = _confidence_color(inv.overall_confidence)
        st.markdown(
            f"""
            <div class="info-card">
                <strong style="font-size:1.05rem">{inv.anomaly.service}</strong>
                &nbsp;&nbsp;<span style="opacity:0.65">{inv.anomaly.date}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Overall confidence:
                <strong style="color:{conf_color_ev}">
                {inv.overall_confidence:.0%}
                </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Hypotheses — 2-column card layout ────────────────────────────────
        st.subheader(f"Hypotheses ({len(inv.hypotheses)})")

        if not inv.hypotheses:
            st.warning(
                "No hypotheses were generated for this anomaly. "
                "This can happen when the LLM cannot establish causal evidence. "
                "Check the raw report in the **Investigation** view for details."
            )
        else:
            for i_h in range(0, len(inv.hypotheses), 2):
                ev_cols = st.columns(2)
                for j_h, ev_col in enumerate(ev_cols):
                    idx_h = i_h + j_h
                    if idx_h >= len(inv.hypotheses):
                        break
                    h = inv.hypotheses[idx_h]
                    conf_color = _confidence_color(h.confidence)
                    ev_count = len(h.evidence) if h.evidence else 0
                    with ev_col:
                        with st.expander(
                            f"#{h.rank}  {h.root_cause[:70]}{'…' if len(h.root_cause) > 70 else ''}",
                            expanded=(h.rank == 1),
                        ):
                            st.markdown(
                                f'<div style="border-left:4px solid {conf_color};padding-left:10px">',
                                unsafe_allow_html=True,
                            )
                            st.progress(h.confidence, text=f"Confidence: {h.confidence:.0%}")
                            st.caption(f"📎 {ev_count} evidence item(s)")
                            if h.evidence:
                                for item in h.evidence:
                                    st.markdown(f"- {item}")
                            if h.cost_calculation:
                                st.markdown("**Cost Calculation:**")
                                st.markdown(
                                    f'<div class="cost-box">{h.cost_calculation}</div>',
                                    unsafe_allow_html=True,
                                )
                            if h.causal_mechanism:
                                st.info(h.causal_mechanism)
                            st.markdown('</div>', unsafe_allow_html=True)

        # ── Ruled-out events — table layout ───────────────────────────────────
        st.subheader(f"Ruled Out Events ({len(inv.ruled_out)})")

        if not inv.ruled_out:
            st.caption("No events were explicitly ruled out in this investigation.")
        else:
            hdr_ev = st.columns([2, 1.8, 3.5])
            hdr_ev[0].markdown("**Event**")
            hdr_ev[1].markdown("**Category**")
            hdr_ev[2].markdown("**Reason**")
            st.markdown(
                "<hr style='margin:4px 0 8px 0;border:none;border-top:1px solid rgba(128,128,128,0.2)'>",
                unsafe_allow_html=True,
            )
            for ro in inv.ruled_out:
                date_str = ro.event_time[:10] if ro.event_time else "N/A"
                row_cols = st.columns([2, 1.8, 3.5])
                row_cols[0].write(f"{ro.event_name}\n\n_{date_str}_")
                row_cols[1].markdown(_category_badge(ro.category), unsafe_allow_html=True)
                row_cols[2].write(ro.reason)
                st.markdown(
                    "<hr style='margin:2px 0;border:none;border-top:1px solid rgba(128,128,128,0.1)'>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — COMPARE INVESTIGATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Compare":
    st.title("⚖️ Compare Investigations")

    invs: list[InvestigationReport] = st.session_state.investigations

    if not st.session_state.data_loaded:
        st.warning("Load data and run at least 2 investigations first.")
    elif len(invs) == 0:
        st.info("No investigations completed yet. Run some from the **Timeline** view.")
    elif len(invs) == 1:
        st.info(
            "Only 1 investigation completed. "
            "Run at least one more to enable comparison."
        )
    else:
        labels = [f"{r.anomaly.service} — {r.anomaly.date}" for r in invs]

        sel_col_a, sel_col_b = st.columns(2)
        with sel_col_a:
            idx_a = st.selectbox(
                "Investigation A",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key="cmp_sel_a",
            )
        with sel_col_b:
            default_b = 1 if len(labels) > 1 else 0
            idx_b = st.selectbox(
                "Investigation B",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key="cmp_sel_b",
                index=default_b,
            )

        if idx_a == idx_b:
            st.warning("Both dropdowns point to the same investigation — select two different ones.")
        else:
            inv_a = invs[idx_a]
            inv_b = invs[idx_b]

            st.divider()

            # ── Side-by-side summary cards ────────────────────────────────────
            col_a, col_b = st.columns(2)

            def _render_summary(inv: InvestigationReport, col) -> None:
                top = inv.hypotheses[0] if inv.hypotheses else None
                root_cause = (
                    (top.root_cause[:85] + "…")
                    if top and len(top.root_cause) > 85
                    else (top.root_cause if top else "—")
                )
                conf_color = _confidence_color(inv.overall_confidence)
                with col:
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div style="font-size:1.05rem;font-weight:700;margin-bottom:8px">
                                {inv.anomaly.service}
                            </div>
                            <p style="margin:2px 0"><b>Date:</b> {inv.anomaly.date}</p>
                            <p style="margin:2px 0">
                                <b>Cost:</b> ${inv.anomaly.cost:.2f}
                                <span style="opacity:0.55">
                                    (expected ${inv.anomaly.expected_cost:.2f})
                                </span>
                            </p>
                            <p style="margin:2px 0">
                                <b>Delta:</b>
                                <span style="color:{RED}">+${inv.anomaly.delta:.2f}</span>
                            </p>
                            <p style="margin:2px 0">
                                <b>Root Cause:</b> {root_cause}
                            </p>
                            <p style="margin:2px 0">
                                <b>Confidence:</b>
                                <span style="color:{conf_color};font-weight:700">
                                {inv.overall_confidence:.0%}
                                </span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            _render_summary(inv_a, col_a)
            _render_summary(inv_b, col_b)

            # ── Metrics comparison table ──────────────────────────────────────
            st.subheader("Metrics Comparison")
            cmp_df = pd.DataFrame(
                {
                    "Metric": [
                        "Overall Confidence",
                        "Evidence Items",
                        "Hypotheses",
                        "Ruled Out",
                        "Elapsed (s)",
                    ],
                    labels[idx_a]: [
                        f"{inv_a.overall_confidence:.0%}",
                        str(sum(len(h.evidence) for h in inv_a.hypotheses)),
                        str(len(inv_a.hypotheses)),
                        str(len(inv_a.ruled_out)),
                        f"{inv_a.elapsed_seconds:.1f}",
                    ],
                    labels[idx_b]: [
                        f"{inv_b.overall_confidence:.0%}",
                        str(sum(len(h.evidence) for h in inv_b.hypotheses)),
                        str(len(inv_b.hypotheses)),
                        str(len(inv_b.ruled_out)),
                        f"{inv_b.elapsed_seconds:.1f}",
                    ],
                }
            )
            st.dataframe(cmp_df, hide_index=True, use_container_width=True)

            # ── Visual comparison grouped bar chart ───────────────────────────
            st.subheader("Visual Comparison")
            ev_a = sum(len(h.evidence) for h in inv_a.hypotheses) if inv_a.hypotheses else 0
            ev_b = sum(len(h.evidence) for h in inv_b.hypotheses) if inv_b.hypotheses else 0

            bar_fig = go.Figure(data=[
                go.Bar(
                    name=labels[idx_a][:30],
                    x=["Confidence", "Evidence Items", "Hypotheses", "Ruled Out"],
                    y=[inv_a.overall_confidence * 100, ev_a, len(inv_a.hypotheses), len(inv_a.ruled_out)],
                    marker_color=BLUE,
                    text=[f"{inv_a.overall_confidence:.0%}", ev_a, len(inv_a.hypotheses), len(inv_a.ruled_out)],
                    textposition="outside",
                ),
                go.Bar(
                    name=labels[idx_b][:30],
                    x=["Confidence", "Evidence Items", "Hypotheses", "Ruled Out"],
                    y=[inv_b.overall_confidence * 100, ev_b, len(inv_b.hypotheses), len(inv_b.ruled_out)],
                    marker_color=GREEN,
                    text=[f"{inv_b.overall_confidence:.0%}", ev_b, len(inv_b.hypotheses), len(inv_b.ruled_out)],
                    textposition="outside",
                ),
            ])
            bar_fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=40, r=40, t=20, b=40),
                yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # ── Recurring pattern detection ───────────────────────────────────
            cats_a = {h.category for h in inv_a.hypotheses}
            cats_b = {h.category for h in inv_b.hypotheses}
            recurring = cats_a & cats_b
            if recurring:
                st.warning(
                    f"⚠️ **Recurring pattern:** category "
                    f"`{'`, `'.join(sorted(recurring))}` appears in both investigations. "
                    "This may indicate a systemic infrastructure issue."
                )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────
elif view == "Feedback":
    st.title("💬 Feedback")

    inv = st.session_state.current_investigation
    if not st.session_state.data_loaded:
        st.warning("Load data and run an investigation first.")
    elif inv is None:
        st.warning("Run an investigation first, then come back here to rate it.")
    else:
        conf_color_fb = _confidence_color(inv.overall_confidence)
        st.markdown(
            f"""
            <div class="info-card">
                <strong>{inv.anomaly.service}</strong>
                &nbsp;·&nbsp; {inv.anomaly.date}
                &nbsp;·&nbsp; Confidence:
                <strong style="color:{conf_color_fb}">{inv.overall_confidence:.0%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        feedback: dict = {
            "anomaly": {
                "service": inv.anomaly.service,
                "date": inv.anomaly.date,
                "z_score": inv.anomaly.z_score,
            },
            "hypotheses": [],
            "overall": "",
            "actual_root_cause": "",
            "timestamp": datetime.now().isoformat(),
        }

        if not inv.hypotheses:
            st.info(
                "This investigation produced no hypotheses to rate. "
                "You can still submit an overall verdict below."
            )
        else:
            st.subheader("Rate Each Hypothesis")
            for h in inv.hypotheses:
                st.markdown(
                    f"**Hypothesis {h.rank}** — "
                    f"{h.root_cause[:110]}{'…' if len(h.root_cause) > 110 else ''}"
                )
                verdict = st.radio(
                    "Verdict",
                    ["Correct ✅", "Incorrect ❌", "Uncertain 🤔"],
                    key=f"fb_verdict_{h.rank}",
                    horizontal=True,
                )
                comment = st.text_input(
                    "Comment (optional)",
                    key=f"fb_comment_{h.rank}",
                    placeholder="e.g. 'Correct, but the instance type was actually m5.xlarge'",
                )
                feedback["hypotheses"].append(
                    {
                        "rank": h.rank,
                        "root_cause": h.root_cause,
                        "verdict": verdict,
                        "comment": comment,
                    }
                )
                st.divider()

        st.subheader("Overall Report Quality")
        overall = st.radio(
            "How would you rate this report?",
            ["Report is actionable", "Report needs corrections", "Report is wrong"],
            key="fb_overall",
        )
        actual_cause = st.text_area(
            "What was the actual root cause? (leave blank if report is correct)",
            key="fb_actual_cause",
            placeholder="Describe what you found when you investigated manually…",
        )

        feedback["overall"] = overall
        feedback["actual_root_cause"] = actual_cause

        if st.button("📤 Submit Feedback", type="primary"):
            try:
                feedback_dir = _ROOT / "data" / "feedback"
                feedback_dir.mkdir(parents=True, exist_ok=True)
                service_slug = (
                    inv.anomaly.service.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                fname = feedback_dir / f"{inv.anomaly.date}_{service_slug}.json"
                fname.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
                st.success(f"✅ Feedback saved → `{fname.relative_to(_ROOT)}`")
            except PermissionError:
                st.error("Permission denied writing to the feedback directory.")
            except Exception as exc:
                st.error(f"Failed to save feedback: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER (rendered on every view)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='cs-footer'>"
    "CostSherlock v1.0 &nbsp;|&nbsp; Vatsal Naik &amp; Priti Ghosh "
    "&nbsp;|&nbsp; Northeastern University &nbsp;|&nbsp; "
    f"Model: <code>{MODEL_NAME}</code>"
    "</div>",
    unsafe_allow_html=True,
)
