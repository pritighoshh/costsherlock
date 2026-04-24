"""Analyst Agent — RAG-powered causal reasoning over cost anomalies."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import anthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents import Anomaly, Hypothesis, RuledOutEvent, SuspectEvent
from rag.retriever import CostSherlockRetriever

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 0.15
MAX_TOKENS = 3000
RETRIEVAL_TOP_K = 8

# Cost-math validation bounds: calculated result must be between
# 20 % and 400 % of the observed delta, otherwise confidence is halved.
CALC_RATIO_MIN = 0.20
CALC_RATIO_MAX = 4.00

_SYSTEM_PROMPT = """\
You are a senior cloud cost forensics expert. You are analyzing a real AWS cost anomaly \
with real CloudTrail evidence.

INPUTS:
- An anomaly: which service spiked, by how much, on what date
- A list of suspect CloudTrail events with timestamps, actors, and resources
- Relevant AWS pricing documentation

YOUR TASK: Identify the root cause with evidence.

ABSOLUTE RULES:
1. EVERY claim must cite a specific CloudTrail event ID or pricing document.
2. You MUST show QUANTITATIVE cost math: units × price = expected cost.
   Then compare to the observed delta. Show the percentage match.
3. CORRELATION ≠ CAUSATION. A temporally close event is NOT automatically the cause.
   You must verify the CAUSAL MECHANISM connects the event type to the pricing dimension.

   Critical example: A PutBucketPolicy event changes ACCESS PERMISSIONS. It does NOT
   affect STORAGE CLASS or STORAGE PRICING. If the anomaly is a storage cost spike,
   a bucket policy change CANNOT be the cause — it must be ruled out as WRONG_MECHANISM.

   Similarly: StopInstances on 2 t3.micro ($0.0104/hr each) cannot explain a $350/day
   EC2 spike — it must be ruled out as WRONG_MAGNITUDE.

4. If no suspect plausibly explains the delta, set root_cause to "INSUFFICIENT_EVIDENCE" and
   confidence to 0.0 — but STILL assign the best-matching category from the valid list based
   on the affected service and observed events. For example: CloudWatch anomaly + alarm events
   → use "logging_misconfiguration". Do NOT use "unknown" unless no valid category fits at all.
5. For every suspect NOT selected, explain why in ruled_out with a category.
6. Assign confidence (0.0-1.0) based on: evidence strength, cost math match, mechanism clarity.

RULING OUT CATEGORIES:
- WRONG_SERVICE: Event affects different service than the anomaly
- WRONG_MAGNITUDE: Event's cost impact is <30% or >300% of observed delta
- WRONG_MECHANISM: Event type doesn't affect the pricing dimension that spiked
- TEMPORAL_ONLY: Near in time but no causal mechanism
- ROUTINE: Known scheduled/automated action with no cost impact

OUTPUT: Valid JSON only, no markdown fences:
{
  "hypotheses": [
    {
      "rank": 1,
      "root_cause": "Clear description of what happened",
      "confidence": 0.85,
      "evidence": ["CloudTrail event [event-id] at [timestamp]: [what it shows]",
                    "Pricing doc [filename]: [relevant pricing fact]"],
      "cost_calculation": "20 instances × $0.34/hr × 24 hrs = $163.20/day vs observed delta $350/day (47% explained by compute alone, likely additional EBS/network costs)",
      "causal_mechanism": "Detailed explanation of WHY this event causes this specific cost increase",
      "category": "compute_overprovisioning"
    }
  ],

VALID CATEGORY VALUES — you MUST use one of these exact strings for the "category" field:
  compute_overprovisioning       — too many/large EC2 instances launched
  compute_misconfiguration       — wrong instance type, AZ, or tenancy selected
  storage_misconfiguration       — wrong S3 storage class, lifecycle, or versioning
  storage_overprovisioning       — excessive EBS volume size or snapshot retention
  database_misconfiguration      — wrong RDS instance class, Multi-AZ, or backup config
  network_misconfiguration       — unexpected NAT Gateway, data transfer, or VPC flow
  logging_misconfiguration       — CloudTrail, CloudWatch Logs, or S3 access-log misconfiguration
  lambda_misconfiguration        — excessive Lambda memory, timeout, or invocation rate
  data_transfer_spike            — unexpected inter-region or internet data transfer
  scheduled_job_runaway          — cron/Auto Scaling action that repeated unexpectedly
  security_event                 — IAM change or policy alteration with cost side-effect
  unknown                        — no category fits; use with INSUFFICIENT_EVIDENCE

  "ruled_out": [
    {
      "event_name": "PutBucketPolicy",
      "event_time": "2026-02-04T16:44:01Z",
      "reason": "Bucket policy changes modify access permissions (who can read/write). They do NOT affect storage class, storage volume, or per-GB pricing. The anomaly is a storage cost increase, which requires a change to lifecycle policies, storage class, or data volume — none of which a policy change affects.",
      "category": "WRONG_MECHANISM"
    }
  ]
}\
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_event_id(suspect: SuspectEvent) -> str:
    """Pull eventID from the raw CloudTrail event, fall back to a stub."""
    return suspect.raw_event.get("eventID", "no-event-id")


def _format_suspect(s: SuspectEvent) -> str:
    """Render a SuspectEvent as a compact, LLM-friendly block."""
    eid = _extract_event_id(s)
    return (
        f"  - EventID: {eid}\n"
        f"    EventName: {s.event_name}\n"
        f"    EventTime: {s.event_time}\n"
        f"    Actor: {s.user_arn}\n"
        f"    Resource: {s.resource_arn}\n"
        f"    ProximityScore: {s.proximity_score:.4f}\n"
        f"    Summary: {s.summary}"
    )


def _format_docs(docs: list[dict]) -> str:
    """Render retrieved RAG chunks as numbered, source-labelled blocks."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"[Doc {i}] Source: {doc['source']}  (relevance={doc['score']:.3f})\n"
            f"{doc['text']}"
        )
    return "\n\n".join(parts)


def _build_user_message(
    anomaly: Anomaly,
    suspects: list[SuspectEvent],
    docs: list[dict],
) -> str:
    """Compose the full user turn sent to the LLM.

    Args:
        anomaly: The detected cost spike to investigate.
        suspects: CloudTrail events ranked by temporal proximity.
        docs: Retrieved RAG chunks (text + source + score).

    Returns:
        Multi-section plain-text message.
    """
    suspect_block = (
        "\n".join(_format_suspect(s) for s in suspects)
        if suspects
        else "  (no suspect events found in the correlation window)"
    )

    return (
        f"## Anomaly Under Investigation\n"
        f"- Service      : {anomaly.service}\n"
        f"- Date         : {anomaly.date}\n"
        f"- Observed Cost: ${anomaly.cost:.2f}\n"
        f"- Expected Cost: ${anomaly.expected_cost:.2f}\n"
        f"- Delta        : ${anomaly.delta:.2f}  ← this is the unexplained increase\n"
        f"- Z-Score      : {anomaly.z_score:.2f}\n"
        f"\n"
        f"## Suspect CloudTrail Events ({len(suspects)} candidates)\n"
        f"{suspect_block}\n"
        f"\n"
        f"## Relevant AWS Pricing Documentation ({len(docs)} chunks)\n"
        f"{_format_docs(docs)}"
    )


def _extract_calc_result(cost_calculation: str) -> Optional[float]:
    """Parse the numeric result of a cost_calculation string.

    Looks for the pattern ``= $N`` that represents the formula output,
    e.g. ``"20 × $0.34 × 24 = $163.20/day"`` → ``163.20``.

    Args:
        cost_calculation: Free-text cost math from the LLM.

    Returns:
        The first dollar amount following an ``=`` sign, or ``None`` if
        no parseable result is found.
    """
    # Match "= $163.20" or "= **$1,620.00**" or "= $163.20/day"
    pattern = r"=\s*\*{0,2}\$([0-9,]+\.?\d*)\*{0,2}"
    matches = re.findall(pattern, cost_calculation)
    if not matches:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def _validate_cost_math(
    hypothesis: Hypothesis,
    delta: float,
) -> Hypothesis:
    """Halve confidence when the cost math result is outside the valid band.

    Valid band: the calculated result must be between 20 % and 400 % of
    *delta*.  If the LLM's arithmetic cannot be parsed the hypothesis is
    returned unchanged (benefit of the doubt).

    Args:
        hypothesis: The Hypothesis to validate (may be mutated).
        delta: The observed cost delta from the anomaly.

    Returns:
        The (potentially updated) hypothesis.
    """
    if delta <= 0:
        return hypothesis

    calc_value = _extract_calc_result(hypothesis.cost_calculation)
    if calc_value is None:
        logger.debug(
            "Could not parse cost_calculation for rank-%d hypothesis — skipping validation",
            hypothesis.rank,
        )
        return hypothesis

    ratio = calc_value / delta
    if not (CALC_RATIO_MIN <= ratio <= CALC_RATIO_MAX):
        original = hypothesis.confidence
        penalised = round(original * 0.5, 4)
        logger.warning(
            "Rank-%d hypothesis cost math out of bounds: calculated=$%.2f, "
            "delta=$%.2f, ratio=%.2f (expected %.2f–%.2f). "
            "Confidence %.2f → %.2f",
            hypothesis.rank,
            calc_value,
            delta,
            ratio,
            CALC_RATIO_MIN,
            CALC_RATIO_MAX,
            original,
            penalised,
        )
        # Return a new instance (Pydantic models are immutable by default)
        return hypothesis.model_copy(update={"confidence": penalised})

    logger.debug(
        "Rank-%d cost math OK: calc=$%.2f, delta=$%.2f, ratio=%.2f",
        hypothesis.rank,
        calc_value,
        delta,
        ratio,
    )
    return hypothesis


def _strip_fences(text: str) -> str:
    """Remove accidental markdown code fences from LLM JSON output."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ── Main agent class ──────────────────────────────────────────────────────────


class Analyst:
    """Agent 3: RAG-powered causal reasoning to explain cost anomalies.

    Uses a local ChromaDB knowledge base (AWS pricing + cost trap docs) to
    ground the LLM's reasoning in factual pricing data, then applies strict
    causal-mechanism and quantitative cost-math validation before accepting
    a hypothesis.

    Attributes:
        _client: Anthropic SDK client.
        _retriever: ChromaDB retriever for the CostSherlock knowledge base.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        """Initialise the Analyst with an Anthropic client and RAG retriever.

        Args:
            db_path: Path to the local ChromaDB vector store built by
                ``rag/ingest.py``.
            anthropic_api_key: Overrides the ``ANTHROPIC_API_KEY`` env var.
                Useful for testing.

        Raises:
            ValueError: If no API key is available.
            ValueError: If the ChromaDB collection does not exist (run
                ``python -m rag.ingest`` first).
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._retriever = CostSherlockRetriever(db_path=db_path)
        logger.info("Analyst initialised — model=%s, top_k=%d", MODEL, RETRIEVAL_TOP_K)

    # ── Core method ──────────────────────────────────────────────────────────

    def analyze(
        self,
        anomaly: Anomaly,
        suspects: list[SuspectEvent],
    ) -> dict:
        """Investigate a cost anomaly and return ranked causal hypotheses.

        Pipeline:
            1. Build a retrieval query from the anomaly's service and delta.
            2. Fetch the top-8 most relevant pricing/troubleshooting chunks.
            3. Call Claude with the system prompt, formatted anomaly, suspects,
               and retrieved docs.
            4. Parse the JSON response into Hypothesis and RuledOutEvent objects.
            5. Validate each hypothesis's cost math against the observed delta.

        Args:
            anomaly: The cost anomaly detected by the Sentinel agent.
            suspects: CloudTrail events ranked by proximity from the Detective.

        Returns:
            Dict with two keys:

            - ``"hypotheses"``: ``list[Hypothesis]`` sorted by rank ascending.
            - ``"ruled_out"``: ``list[RuledOutEvent]``.

        Raises:
            RuntimeError: If the LLM returns unparseable JSON after all retries.
        """
        # Step 1 — retrieval query
        query = f"{anomaly.service} cost increase ${anomaly.delta:.2f}"
        logger.info(
            "Analyst.analyze | service=%s  date=%s  delta=$%.2f  suspects=%d",
            anomaly.service,
            anomaly.date,
            anomaly.delta,
            len(suspects),
        )
        logger.debug("Retrieval query: %r", query)

        # Step 2 — retrieve
        docs = self._retriever.retrieve(query, k=RETRIEVAL_TOP_K)
        logger.info("Retrieved %d chunks for context", len(docs))

        # Step 3 — LLM call (with retry)
        user_msg = _build_user_message(anomaly, suspects, docs)
        raw_json = self._call_llm(user_msg)

        # Step 4 — parse
        hypotheses, ruled_out = self._parse_response(raw_json)

        # Step 5 — validate cost math
        validated: list[Hypothesis] = [
            _validate_cost_math(h, anomaly.delta) for h in hypotheses
        ]
        validated.sort(key=lambda h: h.rank)

        logger.info(
            "Analysis complete: %d hypotheses, %d ruled out",
            len(validated),
            len(ruled_out),
        )
        return {"hypotheses": validated, "ruled_out": ruled_out}

    # ── LLM call (with tenacity retry) ───────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, user_message: str) -> str:
        """Send the user message to Claude and return the raw text response.

        Retried up to 3 times on transient ``anthropic.APIError`` or
        ``anthropic.APIConnectionError`` with exponential back-off (2 s → 4 s
        → 8 s, capped at 30 s).

        Args:
            user_message: The fully-formatted user turn (anomaly + suspects + docs).

        Returns:
            Raw text from the assistant (should be valid JSON).

        Raises:
            anthropic.APIError: If all 3 attempts fail.
        """
        logger.debug("Sending request to %s (max_tokens=%d)", MODEL, MAX_TOKENS)
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text
        logger.debug(
            "LLM responded: %d chars  stop_reason=%s",
            len(text),
            response.stop_reason,
        )
        return text

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(
        self, raw: str
    ) -> tuple[list[Hypothesis], list[RuledOutEvent]]:
        """Parse the LLM's JSON output into typed Pydantic objects.

        Args:
            raw: The raw string returned by the LLM.

        Returns:
            Tuple of (hypotheses, ruled_out).

        Raises:
            RuntimeError: If the JSON cannot be decoded or validated.
        """
        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON:\n%s", cleaned[:500])
            raise RuntimeError(f"Analyst received non-JSON LLM output: {exc}") from exc

        # ── Hypotheses ───────────────────────────────────────────────────────
        hypotheses: list[Hypothesis] = []
        for item in data.get("hypotheses", []):
            try:
                h = Hypothesis(
                    rank=int(item.get("rank", len(hypotheses) + 1)),
                    root_cause=str(item.get("root_cause", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    evidence=list(item.get("evidence", [])),
                    cost_calculation=str(item.get("cost_calculation", "")),
                    causal_mechanism=str(item.get("causal_mechanism", "")),
                    category=str(item.get("category", "unknown")),
                )
                hypotheses.append(h)
                logger.debug(
                    "Hypothesis rank=%d  confidence=%.2f  category=%s",
                    h.rank,
                    h.confidence,
                    h.category,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed hypothesis item: %s — %s", item, exc)

        # ── Ruled out ────────────────────────────────────────────────────────
        ruled_out: list[RuledOutEvent] = []
        for item in data.get("ruled_out", []):
            try:
                r = RuledOutEvent(
                    event_name=str(item.get("event_name", "unknown")),
                    event_time=str(item.get("event_time", "")),
                    reason=str(item.get("reason", "")),
                    category=str(item.get("category", "unknown")),
                )
                ruled_out.append(r)
                logger.debug(
                    "Ruled out: %s  category=%s", r.event_name, r.category
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed ruled_out item: %s — %s", item, exc)

        if not hypotheses:
            logger.warning(
                "LLM returned zero hypotheses — check raw output:\n%s", cleaned[:800]
            )

        return hypotheses, ruled_out


# ── CLI entry point ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Use real synthetic data so the test reflects actual pipeline behaviour.
    from agents.sentinel import Sentinel
    from agents.detective import Detective

    cost_path = "data/synthetic/demo_cost.json"
    log_dir = "data/synthetic/demo_cloudtrail"

    logger.info("Loading synthetic cost data from %s", cost_path)
    df = Sentinel.load_from_json(cost_path)
    anomalies = Sentinel.detect_anomalies(df)

    if not anomalies:
        logger.error("Sentinel found no anomalies in synthetic data — check thresholds")
        sys.exit(1)

    logger.info("Loading CloudTrail logs from %s", log_dir)
    events = Detective.load_cloudtrail_logs(log_dir)

    analyst = Analyst()

    sep = "=" * 70
    for anomaly in anomalies:
        suspects = Detective.get_events_in_window(events, anomaly)

        print(f"\n{sep}")
        print(f"  ANOMALY: {anomaly.service} on {anomaly.date}")
        print(f"  Cost: ${anomaly.cost:.2f}  Expected: ${anomaly.expected_cost:.2f}")
        print(f"  Delta: ${anomaly.delta:.2f}  Z-Score: {anomaly.z_score:.2f}")
        print(f"  Suspects: {len(suspects)}")
        print(sep)

        result = analyst.analyze(anomaly, suspects)

        print(f"\n  HYPOTHESES ({len(result['hypotheses'])}):")
        for h in result["hypotheses"]:
            print(f"\n  [{h.rank}] {h.root_cause}")
            print(f"      Confidence    : {h.confidence:.2f}")
            print(f"      Category      : {h.category}")
            print(f"      Cost Math     : {h.cost_calculation}")
            print(f"      Mechanism     : {h.causal_mechanism[:200]}...")
            print(f"      Evidence ({len(h.evidence)} items):")
            for e in h.evidence:
                print(f"        - {e}")

        print(f"\n  RULED OUT ({len(result['ruled_out'])}):")
        for r in result["ruled_out"]:
            print(f"    [{r.category}] {r.event_name} at {r.event_time}")
            print(f"      {r.reason[:150]}...")

    print(f"\n{sep}\nAnalysis complete.\n")
