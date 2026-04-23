"""Tests for the Detective agent."""

from __future__ import annotations

import pytest

from agents import Anomaly
from agents.detective import Detective, MUTATING_EVENTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEMO_LOG_DIR = "data/synthetic/demo_cloudtrail"

EC2_ANOMALY = Anomaly(
    service="Amazon EC2",
    date="2026-01-30",
    cost=413.72,
    expected_cost=49.50,
    z_score=8.5,
    delta=364.22,
)

S3_ANOMALY = Anomaly(
    service="Amazon S3",
    date="2026-02-04",
    cost=195.45,
    expected_cost=15.80,
    z_score=7.2,
    delta=179.65,
)

CW_ANOMALY = Anomaly(
    service="Amazon CloudWatch",
    date="2026-02-07",
    cost=65.00,
    expected_cost=8.10,
    z_score=4.1,
    delta=56.90,
)


# ---------------------------------------------------------------------------
# load_cloudtrail_logs
# ---------------------------------------------------------------------------

class TestLoadCloudtrailLogs:
    def test_loads_all_files(self) -> None:
        events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        # 5 files: ec2(8) + s3(7) + cloudwatch(4) + rds(3) + noise(12) = 34
        assert len(events) == 34

    def test_handles_records_wrapper_format(self) -> None:
        """ec2_events.json uses {"Records": [...]} — must be unwrapped."""
        events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        ec2_events = [e for e in events if e.get("eventSource") == "ec2.amazonaws.com"]
        assert len(ec2_events) > 0

    def test_handles_plain_list_format(self) -> None:
        """cloudwatch_events.json is a bare JSON array — must be loaded flat.

        4 events from cloudwatch_events.json + 1 DescribeAlarmHistory from noise_events.json
        = 5 events with eventSource monitoring.amazonaws.com.
        """
        events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        cw_events = [e for e in events if e.get("eventSource") == "monitoring.amazonaws.com"]
        assert len(cw_events) == 5  # 4 from cloudwatch_events.json + 1 from noise_events.json

    def test_raises_on_missing_directory(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            Detective.load_cloudtrail_logs(tmp_path / "nonexistent")

    def test_each_event_has_event_name(self) -> None:
        events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        for event in events:
            assert "eventName" in event


# ---------------------------------------------------------------------------
# get_events_in_window — EC2 anomaly
# ---------------------------------------------------------------------------

class TestEC2Suspects:
    @pytest.fixture(autouse=True)
    def load(self) -> None:
        self.events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        self.suspects = Detective.get_events_in_window(self.events, EC2_ANOMALY)

    def test_correct_suspect_count(self) -> None:
        """RunInstances + ModifyInstanceAttribute = 2 suspects."""
        assert len(self.suspects) == 2

    def test_mutating_events_are_included(self) -> None:
        names = {s.event_name for s in self.suspects}
        assert "RunInstances" in names
        assert "ModifyInstanceAttribute" in names

    def test_read_only_events_are_filtered(self) -> None:
        """DescribeInstances, DescribeInstanceStatus, DescribeSecurityGroups must not appear."""
        names = {s.event_name for s in self.suspects}
        assert "DescribeInstances" not in names
        assert "DescribeInstanceStatus" not in names
        assert "DescribeSecurityGroups" not in names

    def test_non_whitelisted_event_filtered(self) -> None:
        """StopInstances is not in MUTATING_EVENTS and must be excluded."""
        assert "StopInstances" not in MUTATING_EVENTS
        names = {s.event_name for s in self.suspects}
        assert "StopInstances" not in names

    def test_out_of_window_event_filtered(self) -> None:
        """TerminateInstances at 2026-01-25 is 5 days before anomaly — outside 48h window."""
        names = {s.event_name for s in self.suspects}
        assert "TerminateInstances" not in names

    def test_proximity_ordering(self) -> None:
        """RunInstances (0.75h before) must rank above ModifyInstanceAttribute (8.33h after)."""
        assert self.suspects[0].event_name == "RunInstances"
        assert self.suspects[1].event_name == "ModifyInstanceAttribute"
        assert self.suspects[0].proximity_score > self.suspects[1].proximity_score

    def test_run_instances_proximity_score(self) -> None:
        """RunInstances at 23:15 on day before → delta=0.75h → score ≈ 0.5714."""
        ri = next(s for s in self.suspects if s.event_name == "RunInstances")
        expected = 1.0 / (1.0 + 0.75)
        assert abs(ri.proximity_score - expected) < 0.001

    def test_modify_instance_attribute_proximity_score(self) -> None:
        """ModifyInstanceAttribute at 08:20 on anomaly day → delta=8.333h → score ≈ 0.1071."""
        mia = next(s for s in self.suspects if s.event_name == "ModifyInstanceAttribute")
        expected = 1.0 / (1.0 + 8.0 + 20.0 / 60.0)
        assert abs(mia.proximity_score - expected) < 0.001

    def test_suspect_has_user_arn(self) -> None:
        ri = next(s for s in self.suspects if s.event_name == "RunInstances")
        assert "deploy-bot" in ri.user_arn

    def test_suspect_summary_is_readable(self) -> None:
        ri = next(s for s in self.suspects if s.event_name == "RunInstances")
        assert "RunInstances" in ri.summary
        assert "c5.2xlarge" in ri.summary

    def test_raw_event_preserved(self) -> None:
        ri = next(s for s in self.suspects if s.event_name == "RunInstances")
        assert ri.raw_event.get("eventName") == "RunInstances"


# ---------------------------------------------------------------------------
# get_events_in_window — S3 anomaly
# ---------------------------------------------------------------------------

class TestS3Suspects:
    @pytest.fixture(autouse=True)
    def load(self) -> None:
        self.events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        self.suspects = Detective.get_events_in_window(self.events, S3_ANOMALY)

    def test_correct_suspect_count(self) -> None:
        """PutBucketLifecycleConfiguration + PutBucketPolicy = 2 suspects."""
        assert len(self.suspects) == 2

    def test_red_herring_included(self) -> None:
        """PutBucketPolicy IS a mutating event — Detective includes it; Analyst rules it out."""
        names = {s.event_name for s in self.suspects}
        assert "PutBucketPolicy" in names

    def test_root_cause_included(self) -> None:
        names = {s.event_name for s in self.suspects}
        assert "PutBucketLifecycleConfiguration" in names

    def test_read_only_events_filtered(self) -> None:
        names = {s.event_name for s in self.suspects}
        assert "GetObject" not in names
        assert "ListObjects" not in names
        assert "GetBucketAcl" not in names
        assert "GetBucketLogging" not in names

    def test_put_object_filtered(self) -> None:
        """PutObject is not in MUTATING_EVENTS whitelist."""
        assert "PutObject" not in MUTATING_EVENTS
        names = {s.event_name for s in self.suspects}
        assert "PutObject" not in names

    def test_out_of_window_filtered(self) -> None:
        """GetBucketAcl at 2026-02-01 is outside the 48h window for 2026-02-04."""
        names = {s.event_name for s in self.suspects}
        assert "GetBucketAcl" not in names

    def test_proximity_ordering(self) -> None:
        """PutBucketPolicy (16:44, 7.27h before) ranks above PutBucketLifecycle (09:15, 14.75h before)."""
        assert self.suspects[0].event_name == "PutBucketPolicy"
        assert self.suspects[1].event_name == "PutBucketLifecycleConfiguration"
        assert self.suspects[0].proximity_score > self.suspects[1].proximity_score

    def test_put_bucket_lifecycle_proximity_score(self) -> None:
        """PutBucketLifecycleConfiguration at 09:15 on Feb 3 → delta=14.75h → score ≈ 0.0634."""
        plc = next(s for s in self.suspects if s.event_name == "PutBucketLifecycleConfiguration")
        expected = 1.0 / (1.0 + 14.75)
        assert abs(plc.proximity_score - expected) < 0.001

    def test_put_bucket_policy_proximity_score(self) -> None:
        """PutBucketPolicy at 16:44 on Feb 3 → delta=7.267h → score ≈ 0.1209."""
        pbp = next(s for s in self.suspects if s.event_name == "PutBucketPolicy")
        delta_hours = 7.0 + 16.0 / 60.0  # 7h 16min = 7.2667h
        expected = 1.0 / (1.0 + delta_hours)
        assert abs(pbp.proximity_score - expected) < 0.001

    def test_bucket_name_in_summary(self) -> None:
        plc = next(s for s in self.suspects if s.event_name == "PutBucketLifecycleConfiguration")
        assert "prod-data-lake" in plc.summary


# ---------------------------------------------------------------------------
# get_events_in_window — CloudWatch anomaly
# ---------------------------------------------------------------------------

class TestCloudWatchSuspects:
    @pytest.fixture(autouse=True)
    def load(self) -> None:
        self.events = Detective.load_cloudtrail_logs(DEMO_LOG_DIR)
        self.suspects = Detective.get_events_in_window(self.events, CW_ANOMALY)

    def test_correct_suspect_count(self) -> None:
        """PutMetricAlarm = 1 suspect."""
        assert len(self.suspects) == 1

    def test_put_metric_alarm_included(self) -> None:
        assert self.suspects[0].event_name == "PutMetricAlarm"

    def test_read_only_events_filtered(self) -> None:
        names = {s.event_name for s in self.suspects}
        assert "DescribeAlarms" not in names
        assert "ListMetrics" not in names
        assert "GetMetricStatistics" not in names

    def test_out_of_window_filtered(self) -> None:
        """GetMetricStatistics at 2026-02-08T11:00 is after window_end (2026-02-07T23:59:59)."""
        names = {s.event_name for s in self.suspects}
        assert "GetMetricStatistics" not in names

    def test_put_metric_alarm_proximity_score(self) -> None:
        """PutMetricAlarm at 14:30 on Feb 6 → delta=9.5h → score = 1/10.5 ≈ 0.09524."""
        pma = self.suspects[0]
        expected = 1.0 / (1.0 + 9.5)
        assert abs(pma.proximity_score - expected) < 0.001

    def test_alarm_name_in_summary(self) -> None:
        assert "high-error-rate-debug" in self.suspects[0].summary


# ---------------------------------------------------------------------------
# MUTATING_EVENTS whitelist
# ---------------------------------------------------------------------------

class TestMutatingEventsWhitelist:
    def test_claude_md_events_present(self) -> None:
        """All events from CLAUDE.md whitelist must be in MUTATING_EVENTS."""
        required = {
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
        }
        assert required.issubset(MUTATING_EVENTS)

    def test_put_metric_alarm_present(self) -> None:
        assert "PutMetricAlarm" in MUTATING_EVENTS

    def test_read_only_names_absent(self) -> None:
        read_only = {"DescribeInstances", "GetObject", "ListBuckets", "DescribeAlarms"}
        assert read_only.isdisjoint(MUTATING_EVENTS)
