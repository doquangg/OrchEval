"""Tests for orcheval.visualization — HTML waterfall visualization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from orcheval.events import NodeEntry, NodeExit
from orcheval.report import report
from orcheval.report.timeline import timeline_report
from orcheval.trace import Trace

if TYPE_CHECKING:
    from pathlib import Path

    from orcheval.events import ErrorEvent, LLMCall

TRACE_ID = "test-trace-0001"


def _extract_json_from_html(html: str) -> dict:
    """Extract the TRACE_DATA JSON blob from the HTML script tag."""
    marker = "const TRACE_DATA = "
    start = html.index(marker) + len(marker)
    # Find the closing semicolon (on its own or before newline)
    end = html.index(";\n", start)
    raw = html[start:end]
    # Reverse the <\/ escaping so json.loads works
    raw = raw.replace(r"<\/", "</")
    return json.loads(raw)


class TestHTMLStructure:
    def test_returns_string(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert isinstance(html, str)

    def test_contains_doctype(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_style_tag(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert "<style>" in html
        assert "</style>" in html

    def test_contains_script_tag(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert "<script>" in html
        assert "</script>" in html

    def test_contains_trace_id(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert TRACE_ID in html

    def test_self_contained(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        # No external stylesheet or script references
        assert '<link rel=' not in html
        assert '<script src=' not in html


class TestHTMLDataEmbed:
    def test_embedded_json_parseable(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        data = _extract_json_from_html(html)
        assert isinstance(data, dict)
        assert "trace_id" in data

    def test_span_count_matches(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        data = _extract_json_from_html(html)
        tl = timeline_report(sample_trace)
        assert len(data["spans"]) == len(tl.spans)

    def test_total_duration_in_data(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        data = _extract_json_from_html(html)
        assert data["total_duration_ms"] == 6000.0

    def test_node_colors_assigned(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        data = _extract_json_from_html(html)
        colors = data["node_colors"]
        assert "agent" in colors
        assert "summarizer" in colors
        # Different nodes get different colors (first-seen order with 12-color palette)
        assert colors["agent"] != colors["summarizer"]

    def test_events_detail_present(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        data = _extract_json_from_html(html)
        assert "events_detail" in data
        # sample_trace has 2 spans
        assert len(data["events_detail"]) == 2


class TestHTMLSummaryPanel:
    def test_contains_duration(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert "6000" in html

    def test_contains_cost(self, sample_trace: Trace) -> None:
        html = sample_trace.to_html()
        assert "0.007" in html

    def test_contains_error_count(
        self, error_retry_events: list[NodeEntry | NodeExit | ErrorEvent | LLMCall],
    ) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        html = trace.to_html()
        data = _extract_json_from_html(html)
        assert data["error_count"] == 2  # SyntaxError + ValueError


class TestHTMLFileOutput:
    def test_writes_to_file(self, sample_trace: Trace, tmp_path: Path) -> None:
        output_path = tmp_path / "trace.html"
        html = sample_trace.to_html(str(output_path))
        assert output_path.exists()
        file_content = output_path.read_text(encoding="utf-8")
        assert file_content == html

    def test_returns_string_regardless(self, sample_trace: Trace, tmp_path: Path) -> None:
        output_path = tmp_path / "trace2.html"
        html_with_path = sample_trace.to_html(str(output_path))
        html_without_path = sample_trace.to_html()
        assert isinstance(html_with_path, str)
        assert isinstance(html_without_path, str)
        # Both should be valid HTML
        assert html_with_path.startswith("<!DOCTYPE html>")
        assert html_without_path.startswith("<!DOCTYPE html>")


class TestHTMLEdgeCases:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        html = trace.to_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        data = _extract_json_from_html(html)
        assert data["spans"] == []

    def test_single_node_trace(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="span-1", node_name="solo"),
            NodeExit(trace_id=TRACE_ID, span_id="span-1", node_name="solo", duration_ms=100.0),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        html = trace.to_html()
        data = _extract_json_from_html(html)
        assert len(data["spans"]) == 1
        assert data["spans"][0]["node_name"] == "solo"

    def test_trace_with_errors(
        self, error_retry_events: list[NodeEntry | NodeExit | ErrorEvent | LLMCall],
    ) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        html = trace.to_html()
        data = _extract_json_from_html(html)
        # Check errors are in events_detail
        has_errors = any(
            detail["errors"]
            for detail in data["events_detail"].values()
        )
        assert has_errors

    def test_state_data_in_detail(self, stateful_trace: Trace) -> None:
        html = stateful_trace.to_html()
        data = _extract_json_from_html(html)
        # Check that state data appears in events_detail
        has_state = any(
            detail["input_state"] or detail["output_state"]
            for detail in data["events_detail"].values()
        )
        assert has_state
        # Check specific state diff
        has_diff = any(
            detail["state_diff"].get("added")
            for detail in data["events_detail"].values()
            if detail["state_diff"]
        )
        assert has_diff


class TestHTMLPrecomputed:
    def test_accepts_precomputed(self, sample_trace: Trace) -> None:
        full = report(sample_trace)
        html = sample_trace.to_html(reports=full)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_matches_without_precomputed(self, sample_trace: Trace) -> None:
        full = report(sample_trace)
        html_with = sample_trace.to_html(reports=full)
        html_without = sample_trace.to_html()
        # Both should embed the same trace data
        data_with = _extract_json_from_html(html_with)
        data_without = _extract_json_from_html(html_without)
        assert data_with["trace_id"] == data_without["trace_id"]
        assert len(data_with["spans"]) == len(data_without["spans"])


class TestHTMLScriptEscaping:
    def test_state_with_script_tag_content(self) -> None:
        """State data containing </script> must not break the HTML."""
        events = [
            NodeEntry(
                trace_id=TRACE_ID, span_id="span-esc", node_name="escaper",
                input_state={"payload": "</script><script>alert(1)</script>"},
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="span-esc", node_name="escaper",
                duration_ms=100.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        html = trace.to_html()
        # The HTML should be valid — the </script> in data should be escaped
        # The script tag should close properly after the JS code
        assert html.count("</script>") == 1  # only the real closing tag
        data = _extract_json_from_html(html)
        assert "</script>" in data["events_detail"]["span-esc"]["input_state"]["payload"]
