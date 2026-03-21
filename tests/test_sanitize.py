"""Tests for orcheval.sanitize — state sanitization and diffing."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from orcheval.sanitize import compute_state_diff, sanitize_state


class TestSanitizeStatePrimitives:
    """Primitive values pass through unchanged."""

    def test_bool(self) -> None:
        assert sanitize_state({"flag": True}) == {"flag": True}

    def test_int(self) -> None:
        assert sanitize_state({"count": 42}) == {"count": 42}

    def test_float(self) -> None:
        assert sanitize_state({"score": 3.14}) == {"score": 3.14}

    def test_none(self) -> None:
        assert sanitize_state({"value": None}) == {"value": None}

    def test_mixed_primitives(self) -> None:
        data = {"a": True, "b": 1, "c": 2.5, "d": None}
        assert sanitize_state(data) == data


class TestSanitizeStateNonDict:
    """Non-dict inputs return empty dict."""

    def test_list_returns_empty(self) -> None:
        assert sanitize_state([1, 2, 3]) == {}

    def test_string_returns_empty(self) -> None:
        assert sanitize_state("hello") == {}

    def test_none_returns_empty(self) -> None:
        assert sanitize_state(None) == {}

    def test_int_returns_empty(self) -> None:
        assert sanitize_state(42) == {}


class TestSanitizeStateStrings:
    """String truncation behavior."""

    def test_short_string_passes_through(self) -> None:
        assert sanitize_state({"s": "hello"}) == {"s": "hello"}

    def test_long_string_truncated(self) -> None:
        long = "x" * 600
        result = sanitize_state({"s": long}, max_string=500)
        assert result["s"] == "x" * 500 + "…[truncated]"

    def test_custom_max_string(self) -> None:
        result = sanitize_state({"s": "abcdefghij"}, max_string=5)
        assert result["s"] == "abcde…[truncated]"

    def test_exact_length_not_truncated(self) -> None:
        result = sanitize_state({"s": "abc"}, max_string=3)
        assert result["s"] == "abc"


class TestSanitizeStatePydantic:
    """Pydantic models are converted via model_dump()."""

    def test_pydantic_model(self) -> None:
        class MyModel(BaseModel):
            name: str
            value: int

        data = {"model": MyModel(name="test", value=42)}
        result = sanitize_state(data)
        # model_dump() produces a dict, which gets serialized as JSON
        assert isinstance(result["model"], (dict, str))


class TestSanitizeStateDataFrame:
    """DataFrame handling (requires pandas)."""

    def test_dataframe_summary(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = sanitize_state({"df": df})
        summary = result["df"]
        assert summary["__type__"] == "DataFrame"
        assert summary["shape"] == [5, 2]
        assert summary["columns"] == ["a", "b"]
        assert len(summary["head"]) == 3


class TestSanitizeStateNumpyArray:
    """Numpy array handling (requires numpy)."""

    def test_ndarray_summary(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.zeros((3, 4))
        result = sanitize_state({"arr": arr})
        summary = result["arr"]
        assert summary["__type__"] == "ndarray"
        assert summary["shape"] == [3, 4]
        assert summary["dtype"] == "float64"


class TestSanitizeStateContainers:
    """List and dict handling."""

    def test_small_list_kept(self) -> None:
        data = {"items": [1, 2, 3]}
        assert sanitize_state(data) == data

    def test_small_dict_kept(self) -> None:
        data = {"nested": {"a": 1, "b": 2}}
        assert sanitize_state(data) == data

    def test_large_list_summarized(self) -> None:
        large = list(range(1000))
        result = sanitize_state({"items": large}, max_json_value=50)
        assert isinstance(result["items"], str)
        assert "1000 items" in result["items"]

    def test_large_dict_summarized(self) -> None:
        large = {f"key_{i}": i for i in range(100)}
        result = sanitize_state({"nested": large}, max_json_value=50)
        assert isinstance(result["nested"], str)
        assert "100 keys" in result["nested"]


class TestSanitizeStateCircularRef:
    """Circular reference detection."""

    def test_circular_dict(self) -> None:
        d: dict[str, Any] = {"a": 1}
        d["self"] = d
        result = sanitize_state({"data": d})
        # The circular ref should be detected, not crash
        assert "data" in result

    def test_circular_list(self) -> None:
        lst: list[Any] = [1, 2]
        lst.append(lst)
        result = sanitize_state({"data": lst})
        assert "data" in result


class TestSanitizeStateBudget:
    """Total size budget enforcement."""

    def test_budget_stops_adding_keys(self) -> None:
        data = {f"key_{i}": "x" * 100 for i in range(100)}
        result = sanitize_state(data, max_size=500)
        # Should have fewer keys than the original
        assert len(result) < 100
        assert len(result) > 0

    def test_zero_budget_returns_some_keys(self) -> None:
        # First key is still added before budget check triggers
        result = sanitize_state({"a": 1, "b": 2, "c": 3}, max_size=0)
        # Budget is checked after first key is added
        assert len(result) <= 1


class TestSanitizeStateNonSerializable:
    """Non-serializable objects use repr() fallback."""

    def test_custom_object_repr(self) -> None:
        class Custom:
            def __repr__(self) -> str:
                return "Custom()"

        result = sanitize_state({"obj": Custom()})
        assert result["obj"] == "Custom()"

    def test_failing_key_skipped(self) -> None:
        class Exploder:
            def __repr__(self) -> str:
                raise RuntimeError("boom")

        # Should not crash — the key is skipped
        result = sanitize_state({"good": 1, "bad": Exploder()})
        assert result.get("good") == 1


class TestSanitizeOutputsCompatibility:
    """Verify refactored _sanitize_outputs matches original behavior."""

    def test_plain_dict_with_primitives(self) -> None:
        data = {"flag": True, "count": 42, "score": 3.14, "empty": None}
        result = sanitize_state(data, max_size=2000, max_string=200, max_json_value=500)
        assert result == data

    def test_string_truncation_at_200(self) -> None:
        long = "x" * 300
        result = sanitize_state({"s": long}, max_size=2000, max_string=200, max_json_value=500)
        assert result["s"] == "x" * 200 + "…[truncated]"

    def test_small_json_value_kept(self) -> None:
        data = {"items": [1, 2, 3]}
        result = sanitize_state(data, max_size=2000, max_string=200, max_json_value=500)
        assert result == data

    def test_large_json_value_summarized(self) -> None:
        data = {"items": list(range(200))}  # serializes to >500 chars
        result = sanitize_state(data, max_size=2000, max_string=200, max_json_value=500)
        assert isinstance(result["items"], str)

    def test_non_dict_returns_empty(self) -> None:
        result = sanitize_state("not a dict", max_size=2000, max_string=200, max_json_value=500)
        assert result == {}

    def test_budget_enforcement(self) -> None:
        data = {f"key_{i}": "x" * 150 for i in range(30)}
        result = sanitize_state(data, max_size=2000, max_string=200, max_json_value=500)
        total_chars = sum(len(str(v)) for v in result.values())
        # Should respect the 2000-char budget (approximately)
        assert total_chars <= 2500  # some slack for key overhead


class TestComputeStateDiff:
    """State diff computation."""

    def test_no_changes(self) -> None:
        state = {"a": 1, "b": "hello"}
        diff = compute_state_diff(state, state)
        assert diff == {"added": [], "removed": [], "modified": []}

    def test_added_keys(self) -> None:
        entry = {"a": 1}
        exit_ = {"a": 1, "b": 2, "c": 3}
        diff = compute_state_diff(entry, exit_)
        assert diff["added"] == ["b", "c"]
        assert diff["removed"] == []
        assert diff["modified"] == []

    def test_removed_keys(self) -> None:
        entry = {"a": 1, "b": 2}
        exit_ = {"a": 1}
        diff = compute_state_diff(entry, exit_)
        assert diff["added"] == []
        assert diff["removed"] == ["b"]
        assert diff["modified"] == []

    def test_modified_keys(self) -> None:
        entry = {"a": 1, "b": "old"}
        exit_ = {"a": 1, "b": "new"}
        diff = compute_state_diff(entry, exit_)
        assert diff["added"] == []
        assert diff["removed"] == []
        assert diff["modified"] == ["b"]

    def test_mixed_changes(self) -> None:
        entry = {"a": 1, "b": 2, "c": 3}
        exit_ = {"a": 1, "b": 99, "d": 4}
        diff = compute_state_diff(entry, exit_)
        assert diff["added"] == ["d"]
        assert diff["removed"] == ["c"]
        assert diff["modified"] == ["b"]

    def test_empty_states(self) -> None:
        diff = compute_state_diff({}, {})
        assert diff == {"added": [], "removed": [], "modified": []}

    def test_comparison_error_treated_as_modified(self) -> None:
        """If != raises, the key is treated as modified."""

        class Uncomparable:
            def __eq__(self, other: object) -> bool:
                raise TypeError("cannot compare")

        entry = {"x": Uncomparable()}
        exit_ = {"x": Uncomparable()}
        diff = compute_state_diff(entry, exit_)
        assert diff["modified"] == ["x"]

    def test_keys_sorted(self) -> None:
        entry = {"c": 1, "a": 2}
        exit_ = {"b": 3, "c": 99}
        diff = compute_state_diff(entry, exit_)
        assert diff["added"] == ["b"]
        assert diff["removed"] == ["a"]
        assert diff["modified"] == ["c"]
