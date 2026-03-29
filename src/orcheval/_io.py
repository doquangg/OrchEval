"""Shared output-path resolution and file-writing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = "orcheval_outputs"


def resolve_output_path(path: str) -> Path:
    """Resolve an output path, prepending the default output dir for bare filenames."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        # Bare filename like "trace.html" -> orcheval_outputs/trace.html
        return Path(DEFAULT_OUTPUT_DIR) / p
    return p


def json_safe(obj: Any) -> Any:
    """Round-trip through JSON to coerce non-primitive types (numpy, pandas, etc.)."""
    return json.loads(json.dumps(obj, default=str))


def write_output(content: str, path: str) -> None:
    """Resolve *path* and write *content* to it, creating directories as needed."""
    resolved = resolve_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
