"""State sanitization utilities for converting arbitrary Python objects to JSON-safe dicts.

Used by adapters to capture graph state at node boundaries and for
decision context in routing inference.
"""

from __future__ import annotations

import json
from typing import Any


def sanitize_state(
    data: Any,
    *,
    max_size: int = 10_000,
    max_string: int = 500,
    max_json_value: int = 2_000,
) -> dict[str, Any]:
    """Convert arbitrary Python data into a JSON-safe dict.

    Handles Pydantic models, DataFrames, numpy arrays, circular references,
    and non-serializable objects gracefully.  Preserves dict structure (key names
    and types) even when individual values are truncated.

    Args:
        data: The data to sanitize. Non-dict inputs return ``{}``.
        max_size: Total character budget for the output. Keys stop being added
            once the budget is exhausted.
        max_string: Maximum characters for individual string values.
        max_json_value: Maximum serialized characters for list/dict values
            before they are summarized instead of kept verbatim.

    Returns:
        A JSON-safe dict suitable for embedding in events.
    """
    if not isinstance(data, dict):
        return {}

    safe: dict[str, Any] = {}
    budget = max_size
    seen: set[int] = set()

    for key, value in data.items():
        if budget <= 0:
            break
        try:
            sanitized, cost = _sanitize_value(
                value, max_string=max_string, max_json_value=max_json_value, seen=seen
            )
            safe[key] = sanitized
            budget -= cost
        except Exception:
            # Never crash on a single key — skip it
            pass

    return safe


def _sanitize_value(
    value: Any,
    *,
    max_string: int,
    max_json_value: int,
    seen: set[int],
) -> tuple[Any, int]:
    """Sanitize a single value, returning (sanitized_value, char_cost).

    The char_cost is a rough estimate used for budget tracking.
    """
    # Circular reference detection for mutable containers
    if isinstance(value, (dict, list, set)) and id(value) in seen:
        sentinel = "<circular ref>"
        return sentinel, len(sentinel)
    if isinstance(value, (dict, list, set)):
        seen.add(id(value))

    # Primitives pass through
    if isinstance(value, (bool, int, float, type(None))):
        return value, len(str(value))

    # Strings — truncate if too long
    if isinstance(value, str):
        if len(value) > max_string:
            truncated = value[:max_string] + "…[truncated]"
            return truncated, len(truncated)
        return value, len(value)

    # Pydantic BaseModel — convert to dict then recurse
    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            dumped = value.model_dump()
            return _sanitize_container(
                dumped, max_string=max_string, max_json_value=max_json_value, seen=seen
            )
    except ImportError:
        pass

    # numpy arrays — shape + dtype summary
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            summary: dict[str, Any] = {
                "__type__": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            rep = json.dumps(summary)
            return summary, len(rep)
    except ImportError:
        pass

    # pandas DataFrames — shape + columns + head preview
    try:
        import pandas as pd  # type: ignore[import-untyped]

        if isinstance(value, pd.DataFrame):
            summary_df: dict[str, Any] = {
                "__type__": "DataFrame",
                "shape": list(value.shape),
                "columns": list(value.columns),
                "head": value.head(3).to_dict(orient="records"),
            }
            rep_df = json.dumps(summary_df, default=str)
            return summary_df, len(rep_df)
    except ImportError:
        pass

    # Lists and dicts — serialize; keep if under budget, else summarize
    if isinstance(value, (list, dict)):
        return _sanitize_container(
            value, max_string=max_string, max_json_value=max_json_value, seen=seen
        )

    # Fallback: repr() for anything else
    try:
        rep_str = repr(value)
        if len(rep_str) > max_string:
            rep_str = rep_str[:max_string] + "…[truncated]"
        return rep_str, len(rep_str)
    except Exception:
        fallback = f"<unrepresentable {type(value).__name__}>"
        return fallback, len(fallback)


def _sanitize_container(
    value: list[Any] | dict[str, Any],
    *,
    max_string: int,
    max_json_value: int,
    seen: set[int],
) -> tuple[Any, int]:
    """Sanitize a list or dict, keeping it verbatim if small enough."""
    try:
        serialized = json.dumps(value, default=str)
        if len(serialized) <= max_json_value:
            return value, len(serialized)
    except (TypeError, ValueError):
        pass

    # Over budget or not serializable — summarize
    if isinstance(value, dict):
        summary = f"<dict with {len(value)} keys: {list(value.keys())[:10]}>"
        if len(summary) > max_string:
            summary = summary[:max_string] + "…[truncated]"
        return summary, len(summary)
    else:
        summary = f"<list with {len(value)} items>"
        return summary, len(summary)


def compute_state_diff(
    entry_state: dict[str, Any],
    exit_state: dict[str, Any],
) -> dict[str, list[str]]:
    """Compute top-level key differences between entry and exit state.

    Returns a dict with ``"added"``, ``"removed"``, and ``"modified"`` lists.

    Note: Uses ``!=`` for value comparison, which is safe here because both
    inputs are sanitized (size-budgeted) dicts — deep comparisons are bounded.
    """
    entry_keys = set(entry_state.keys())
    exit_keys = set(exit_state.keys())

    added = sorted(exit_keys - entry_keys)
    removed = sorted(entry_keys - exit_keys)

    modified: list[str] = []
    for key in sorted(entry_keys & exit_keys):
        try:
            if entry_state[key] != exit_state[key]:
                modified.append(key)
        except Exception:
            # If comparison raises (e.g. uncomparable types), assume modified
            modified.append(key)

    return {"added": added, "removed": removed, "modified": modified}
