"""DataFrame export for orchestration traces.

Produces a pandas DataFrame with one row per event, flattening
event-specific fields into columns.  Pandas is a lazy optional
dependency — imported at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orcheval.trace import Trace

# Columns extracted via getattr on the event object.
# These are real Pydantic model attributes on one or more event subclasses.
_COMMON_COLUMNS = (
    "event_type",
    "timestamp",
    "trace_id",
    "node_name",
    "span_id",
    "parent_span_id",
)

_ATTR_COLUMNS = (
    "duration_ms",
    "model",
    "input_tokens",
    "output_tokens",
    "cost",
    "tool_name",
    "error_type",
    "error_message",
    "source_node",
    "target_node",
    "sender",
    "receiver",
    "content_summary",
    "pass_number",
    "direction",
)

ALL_COLUMNS = (*_COMMON_COLUMNS, *_ATTR_COLUMNS, "has_state_data")


def build_dataframe(trace: Trace) -> Any:
    """Build a pandas DataFrame from a trace (one row per event).

    Returns a ``pandas.DataFrame``.  Raises ``ImportError`` with an
    install hint if pandas is not available.
    """
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install orcheval[pandas]"
        ) from None

    rows: list[dict[str, Any]] = []
    for event in trace:
        # Strategy 1: real event attributes
        row: dict[str, Any] = {col: getattr(event, col, None) for col in _COMMON_COLUMNS}
        for col in _ATTR_COLUMNS:
            row[col] = getattr(event, col, None)

        # Strategy 2: derived field — NOT an attribute on any event class
        row["has_state_data"] = bool(
            getattr(event, "input_state", None)
        ) or bool(
            getattr(event, "output_state", None)
        )

        rows.append(row)

    return pd.DataFrame(rows, columns=list(ALL_COLUMNS))
