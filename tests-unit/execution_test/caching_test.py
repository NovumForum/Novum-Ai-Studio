import sys
from unittest.mock import MagicMock

sys.modules["psutil"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["nodes"] = MagicMock()

from comfy_execution.caching import to_hashable, Unhashable

from enum import Enum

class SampleStrEnum(str, Enum):
    VAL = "test_enum"

def test_to_hashable_primitives():
    assert to_hashable(123) == 123
    assert to_hashable(3.14) == 3.14
    assert to_hashable("hello") == "hello"
    assert to_hashable(True) is True
    assert to_hashable(False) is False
    assert to_hashable(b"bytes") == b"bytes"
    assert to_hashable(None) is None
    assert to_hashable(SampleStrEnum.VAL) == SampleStrEnum.VAL

def test_to_hashable_sequences():
    data_list = [1, "a", True]
    data_tuple = (1, "a", True)
    assert to_hashable(data_list) == (1, "a", True)
    assert to_hashable(data_tuple) == (1, "a", True)

def test_to_hashable_mappings():
    data_dict = {"b": 2, "a": 1}
    # Keys should be sorted and returned as tuple of pairs
    assert to_hashable(data_dict) == (("a", 1), ("b", 2))

def test_to_hashable_nested():
    nested = {
        "node_id": 1,
        "inputs": ["str", [10, 20], {"x": 1.0}],
    }
    expected = (
        ("inputs", ("str", (10, 20), (("x", 1.0),))),
        ("node_id", 1),
    )
    assert to_hashable(nested) == expected

def test_to_hashable_unhashable():
    class CustomObj:
        pass

    result = to_hashable(CustomObj())
    assert isinstance(result, Unhashable)
