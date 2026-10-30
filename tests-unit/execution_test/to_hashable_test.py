import sys
import unittest.mock as mock
from collections.abc import Mapping, Sequence

# Mock modules if missing in test environment before importing caching
for mod in ["psutil", "torch", "nodes", "typing_extensions"]:
    if mod not in sys.modules:
        sys.modules[mod] = mock.MagicMock()

from comfy_execution.caching import to_hashable, Unhashable


class CustomMapping(Mapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class CustomSequence(Sequence):
    def __init__(self, items):
        self._items = items

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


def test_to_hashable_primitives():
    assert to_hashable(42) == 42
    assert to_hashable(3.14) == 3.14
    assert to_hashable("hello") == "hello"
    assert to_hashable(True) is True
    assert to_hashable(None) is None
    assert to_hashable(b"binary") == b"binary"


def test_to_hashable_dict():
    d = {"b": 2, "a": 1}
    result = to_hashable(d)
    assert isinstance(result, frozenset)
    assert result == frozenset([("a", 1), ("b", 2)])


def test_to_hashable_list_and_tuple():
    lst = [10, "x", True]
    tpl = (10, "x", True)
    res_lst = to_hashable(lst)
    res_tpl = to_hashable(tpl)
    expected = frozenset([(0, 10), (1, "x"), (2, True)])
    assert res_lst == expected
    assert res_tpl == expected


def test_to_hashable_nested():
    nested = {
        "node_1": ["KSampler", {"seed": 12345, "cfg": 7.5}],
        "node_2": ("CLIPTextEncode", {"text": "prompt"}),
    }
    result = to_hashable(nested)
    assert isinstance(result, frozenset)


def test_to_hashable_custom_mapping_and_sequence():
    cm = CustomMapping({"k1": "v1"})
    res_cm = to_hashable(cm)
    assert res_cm == frozenset([("k1", "v1")])

    cs = CustomSequence(["item1", "item2"])
    res_cs = to_hashable(cs)
    assert res_cs == frozenset([(0, "item1"), (1, "item2")])


def test_to_hashable_unhashable():
    class DummyObj:
        pass

    result = to_hashable(DummyObj())
    assert isinstance(result, Unhashable)
