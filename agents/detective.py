"""Detective Agent — correlates CloudTrail events with detected anomalies."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents import Anomaly, SuspectEvent

logger = logging.getLogger(__name__)

# Canonical whitelist from CLAUDE.md + PutMetricAlarm (CloudWatch config mutation)
MUTATING_EVENTS: frozenset[str] = frozenset({
    "RunInstances",
    "TerminateInstances",
    "ModifyDBInstance",
    "PutBucketLifecycleConfiguration",
    "PutBucketPolicy",
    "CreateFunction20150331",
    "UpdateFunctionConfiguration",
    "CreateNatGateway",
    "ModifyInstanceAttribute",
    "CreateAutoScalingGroup",
    "PutMetricAlarm",
})


class Detective:
    """Agent 2: Correlates CloudTrail events with cost anomalies.

    Loads CloudTrail logs from a directory, filters to mutating events within
    a time window before the anomaly, and ranks them by temporal proximity.
    """

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_cloudtrail_logs(log_dir: str | Path) -> list[dict]:
        """Load all CloudTrail log files from a directory.

        Handles both single-event dicts and ``{"Records": [...]}`` format.

        Args:
            log_dir: Directory containing ``.json`` CloudTrail log files.

        Returns:
            Flat list of individual CloudTrail event dicts.

        Raises:
            FileNotFoundError: If *log_dir* does not exist.
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(
                f"CloudTrail log directory not found: {log_dir}"
            )

        events: list[dict] = []
        for json_file in sorted(log_dir.glob("*.json")):
            with json_file.open() as fh:
                data = json.load(fh)

            if isinstance(data, list):
                events.extend(data)
            elif isinstance(data, dict) and "Records" in data:
                events.extend(data["Records"])
            elif isinstance(data, dict):
                events.append(data)
            else:
                logger.warning("Unexpected JSON structure in %s — skipping", json_file.name)
                continue

            logger.debug("Loaded %s", json_file.name)

        logger.info(
            "Loaded %d CloudTrail events from %d files in %s",
            len(events),
            sum(1 for _ in log_dir.glob("*.json")),
            log_dir,
        )
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_event_time(event: dict) -> datetime | None:
        """Parse eventTime from a CloudTrail event dict.

        Args:
            event: A single CloudTrail event dict.

        Returns:
            Timezone-aware datetime, or ``None`` if unparseable.
        """
        raw = event.get("eventTime")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse eventTime: %r", raw)
            return None

    @staticmethod
    def _build_summary(event: dict) -> str:
        """Generate a human-readable summary for a CloudTrail event.

        Args:
            event: A single CloudTrail event dict.

        Returns:
            One-line English description of the event.
        """
        name = event.get("eventName", "UnknownEvent")
        identity = event.get("userIdentity", {})
        user_arn = identity.get("arn", "unknown")
        # Use the last segment of the ARN as a short name
        actor = user_arn.split("/")[-1] if "/" in user_arn else user_arn
        region = event.get("awsRegion", "us-east-1")
        source_ip = event.get("sourceIPAddress", "unknown")
        params = event.get("requestParameters") or {}

        if name == "RunInstances":
            count = params.get("maxCount", 1)
            itype = params.get("instanceType", "unknown")
            return (
                f"RunInstances: {count}x {itype} launched by {actor} "
                f"in {region} from {source_ip}"
            )
        if name == "PutBucketLifecycleConfiguration":
            bucket = params.get("bucketName", "unknown-bucket")
            return (
                f"PutBucketLifecycleConfiguration: lifecycle rules updated on "
                f"bucket '{bucket}' by {actor} from {source_ip}"
            )
        if name == "PutBucketPolicy":
            bucket = params.get("bucketName", "unknown-bucket")
            return (
                f"PutBucketPolicy: bucket policy replaced on '{bucket}' "
                f"by {actor} from {source_ip}"
            )
        if name == "ModifyInstanceAttribute":
            attr = params.get("attribute", "unknown")
            return (
                f"ModifyInstanceAttribute: attribute '{attr}' modified "
                f"by {actor} in {region} from {source_ip}"
            )
        if name == "PutMetricAlarm":
            alarm = params.get("alarmName", "unknown-alarm")
            return (
                f"PutMetricAlarm: alarm '{alarm}' created/updated "
                f"by {actor} in {region} from {source_ip}"
            )
        if name == "TerminateInstances":
            return (
                f"TerminateInstances: instances terminated "
                f"by {actor} in {region} from {source_ip}"
            )

        return f"{name} by {actor} in {region} from {source_ip}"

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    @classmethod
    def get_events_in_window(
        cls,
        events: list[dict],
        anomaly: Anomaly,
        hours_before: int = 48,
    ) -> list[SuspectEvent]:
        """Filter and rank CloudTrail events correlated with an anomaly.

        Keeps only mutating events (``MUTATING_EVENTS`` whitelist) that fall
        within the window ``[anomaly_date - hours_before, end_of_anomaly_date]``.
        Each event is scored by::

            proximity_score = 1 / (1 + hours_delta)

        where ``hours_delta`` is the absolute difference in hours between the
        event time and the start of the anomaly date.

        Args:
            events: Flat list of CloudTrail events from :meth:`load_cloudtrail_logs`.
            anomaly: The detected anomaly to correlate against.
            hours_before: Window size in hours before the anomaly date (default 48).

        Returns:
            List of :class:`~agents.SuspectEvent` sorted by ``proximity_score``
            descending (highest = closest to anomaly).
        """
        anomaly_dt = datetime.fromisoformat(anomaly.date).replace(tzinfo=timezone.utc)
        window_start = anomaly_dt - timedelta(hours=hours_before)
        window_end = anomaly_dt + timedelta(hours=24)  # include full anomaly day

        suspects: list[SuspectEvent] = []

        for event in events:
            event_name = event.get("eventName", "")
            if event_name not in MUTATING_EVENTS:
                continue

            event_dt = cls._parse_event_time(event)
            if event_dt is None:
                continue

            if not (window_start <= event_dt <= window_end):
                continue

            hours_delta = abs((event_dt - anomaly_dt).total_seconds() / 3600.0)
            proximity_score = 1.0 / (1.0 + hours_delta)

            user_arn = event.get("userIdentity", {}).get(
                "arn", "arn:aws:iam::123456789012:user/unknown"
            )

            resources = event.get("resources", [])
            resource_arn = (
                resources[0].get("ARN", "arn:aws:*::123456789012:unknown")
                if resources
                else "arn:aws:*::123456789012:unknown"
            )

            suspects.append(
                SuspectEvent(
                    event_name=event_name,
                    event_time=event.get("eventTime", ""),
                    user_arn=user_arn,
                    resource_arn=resource_arn,
                    proximity_score=round(proximity_score, 6),
                    summary=cls._build_summary(event),
                    raw_event=event,
                )
            )
            logger.debug(
                "Suspect: %s at %s (score=%.4f)",
                event_name,
                event.get("eventTime"),
                proximity_score,
            )

        suspects.sort(key=lambda s: s.proximity_score, reverse=True)
        logger.info(
            "Found %d suspect events for %s anomaly on %s",
            len(suspects),
            anomaly.service,
            anomaly.date,
        )
        return suspects
