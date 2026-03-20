"""Error pattern analysis and retry behavior detection."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from orcheval.events import ErrorEvent, NodeEntry, NodeExit

if TYPE_CHECKING:
    from orcheval.trace import Trace

UNKNOWN_NODE = "<unknown>"


class ErrorCluster(BaseModel):
    """A group of errors sharing the same error_type."""

    model_config = {"frozen": True}

    error_type: str
    count: int
    messages: list[str] = Field(default_factory=list)
    nodes: list[str] = Field(default_factory=list)
    first_occurrence_offset_ms: float | None = None
    last_occurrence_offset_ms: float | None = None


class RetrySequence(BaseModel):
    """A detected sequence of retries for a single node."""

    model_config = {"frozen": True}

    node_name: str
    attempt_count: int
    succeeded: bool
    errors: list[str] = Field(default_factory=list)
    total_retry_duration_ms: float | None = None


class RetryReport(BaseModel):
    """Full error and retry analysis."""

    model_config = {"frozen": True}

    error_clusters: list[ErrorCluster] = Field(default_factory=list)
    retry_sequences: list[RetrySequence] = Field(default_factory=list)
    total_errors: int = 0
    unique_error_types: int = 0
    overall_retry_success_rate: float | None = None
    nodes_with_errors: list[str] = Field(default_factory=list)


def _build_error_clusters(
    trace: Trace,
    start_ts: Any,
) -> list[ErrorCluster]:
    """Group errors by error_type with deduplication."""
    errors = trace.get_events_by_type(ErrorEvent)
    if not errors:
        return []

    groups: dict[str, list[ErrorEvent]] = defaultdict(list)
    for e in errors:
        groups[e.error_type].append(e)

    clusters = []
    for error_type, events in sorted(groups.items()):
        messages = sorted(set(e.error_message for e in events))
        nodes = sorted(set(
            e.node_name or UNKNOWN_NODE for e in events
        ))

        first_offset: float | None = None
        last_offset: float | None = None
        if start_ts is not None:
            first_offset = (events[0].timestamp - start_ts).total_seconds() * 1000
            last_offset = (events[-1].timestamp - start_ts).total_seconds() * 1000

        clusters.append(ErrorCluster(
            error_type=error_type,
            count=len(events),
            messages=messages,
            nodes=nodes,
            first_occurrence_offset_ms=first_offset,
            last_occurrence_offset_ms=last_offset,
        ))
    return clusters


def _detect_retry_sequences(trace: Trace) -> list[RetrySequence]:
    """Walk events chronologically to detect retry patterns per node.

    A retry is detected when a node has multiple NodeEntry events,
    with ErrorEvents between them. The sequence tracks whether the
    final attempt succeeded (ended with NodeExit without error).
    """
    timeline = trace.get_timeline()
    if not timeline:
        return []

    # Track per-node state
    # For each node, collect (entry_timestamp, errors_since_entry, exited_clean)
    node_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Current attempt state per node
    current_attempt: dict[str, dict[str, Any]] = {}

    for event in timeline:
        if isinstance(event, NodeEntry):
            node = event.node_name
            # If there's a previous attempt with errors, it was a failed attempt
            if node in current_attempt:
                # Previous attempt exists — this is a retry
                pass
            current_attempt[node] = {
                "entry_ts": event.timestamp,
                "errors": [],
                "exited_clean": False,
            }
            node_entries[node].append(current_attempt[node])

        elif isinstance(event, ErrorEvent) and event.node_name:
            node = event.node_name
            if node in current_attempt:
                current_attempt[node]["errors"].append(event.error_message)

        elif isinstance(event, NodeExit):
            node = event.node_name
            if node in current_attempt:
                attempt = current_attempt[node]
                if not attempt["errors"]:
                    attempt["exited_clean"] = True
                attempt["exit_ts"] = event.timestamp

    # Build retry sequences: only for nodes with >1 entry (indicating retries)
    sequences = []
    for node, attempts in sorted(node_entries.items()):
        if len(attempts) < 2:
            continue

        # Check if any attempt had errors (otherwise multiple entries aren't retries)
        has_errors = any(a["errors"] for a in attempts)
        if not has_errors:
            continue

        all_errors = []
        for a in attempts:
            all_errors.extend(a["errors"])

        # Succeeded if the last attempt exited cleanly
        succeeded = attempts[-1]["exited_clean"]

        # Duration from first entry to last exit (or last entry if no exit)
        first_ts = attempts[0]["entry_ts"]
        last_ts = attempts[-1].get("exit_ts", attempts[-1]["entry_ts"])
        duration = (last_ts - first_ts).total_seconds() * 1000

        sequences.append(RetrySequence(
            node_name=node,
            attempt_count=len(attempts),
            succeeded=succeeded,
            errors=all_errors,
            total_retry_duration_ms=duration,
        ))

    return sequences


def retry_report(trace: Trace) -> RetryReport:
    """Analyze error patterns and retry behavior."""
    errors = trace.get_events_by_type(ErrorEvent)
    if not errors:
        return RetryReport()

    timeline = trace.get_timeline()
    start_ts = timeline[0].timestamp if timeline else None

    clusters = _build_error_clusters(trace, start_ts)
    sequences = _detect_retry_sequences(trace)

    nodes_with_errors = sorted(set(
        e.node_name or UNKNOWN_NODE for e in errors
    ))

    success_rate: float | None = None
    if sequences:
        succeeded = sum(1 for s in sequences if s.succeeded)
        success_rate = succeeded / len(sequences)

    return RetryReport(
        error_clusters=clusters,
        retry_sequences=sequences,
        total_errors=len(errors),
        unique_error_types=len(clusters),
        overall_retry_success_rate=success_rate,
        nodes_with_errors=nodes_with_errors,
    )
