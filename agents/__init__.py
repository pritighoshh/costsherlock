"""Inter-agent data contracts for CostSherlock."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Anomaly(BaseModel):
    """A detected cost anomaly for a given AWS service and date."""

    service: str
    date: str
    cost: float
    expected_cost: float
    z_score: float
    delta: float


class SuspectEvent(BaseModel):
    """A CloudTrail event correlated with a cost anomaly."""

    event_name: str
    event_time: str
    user_arn: str
    resource_arn: str
    proximity_score: float
    summary: str
    raw_event: dict = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """A ranked causal hypothesis produced by the Analyst agent."""

    rank: int
    root_cause: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    cost_calculation: str
    causal_mechanism: str
    category: str


class RuledOutEvent(BaseModel):
    """A CloudTrail event considered and ruled out as a cause."""

    event_name: str
    event_time: str
    reason: str
    category: str


class InvestigationReport(BaseModel):
    """Final structured output combining all agent findings."""

    anomaly: Anomaly
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    ruled_out: list[RuledOutEvent] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    overall_confidence: float
    report_markdown: str = ""
    elapsed_seconds: float = 0.0
