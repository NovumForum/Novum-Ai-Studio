import sys
from unittest.mock import MagicMock

# Mock non-unit test dependencies when running standalone caching test
sys.modules.setdefault("psutil", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("nodes", MagicMock())

from comfy_execution.caching import to_hashable, Unhashable, PRIMITIVE_TYPES


def test_to_hashable_primitives():
    assert to_hashable(123) == 123
    assert to_hashable(3.14) == 3.14
    assert to_hashable("test_string") == "test_string"
    assert to_hashable(True) is True
    assert to_hashable(False) is False
    assert to_hashable(b"bytes_data") == b"bytes_data"
    assert to_hashable(None) is None


def test_to_hashable_sequences():
    lst = [1, "a", [2, 3]]
    h_lst = to_hashable(lst)
    assert h_lst == (1, "a", (2, 3))
    assert isinstance(h_lst, tuple)

    tpl = (1, ("b", 4))
    h_tpl = to_hashable(tpl)
    assert h_tpl == (1, ("b", 4))
    assert isinstance(h_tpl, tuple)


def test_to_hashable_mappings():
    d1 = {"a": 1, "b": [1, 2], "c": {"d": 3}}
    d2 = {"c": {"d": 3}, "b": [1, 2], "a": 1}

    h1 = to_hashable(d1)
    h2 = to_hashable(d2)

    assert isinstance(h1, frozenset)
    assert isinstance(h2, frozenset)
    assert h1 == h2
    assert hash(h1) == hash(h2)


def test_to_hashable_sets():
    s1 = {1, 2, 3}
    s2 = {3, 2, 1}

    h1 = to_hashable(s1)
    h2 = to_hashable(s2)

    assert isinstance(h1, frozenset)
    assert isinstance(h2, frozenset)
    assert h1 == h2
    assert hash(h1) == hash(h2)


def test_to_hashable_unhashable():
    class DummyCustomObject:
        pass

    obj = DummyCustomObject()
    h_obj = to_hashable(obj)
    assert isinstance(h_obj, Unhashable)


def test_to_hashable_node_signature_structure():
    sig1 = [
        "KSampler",
        0,
        ("cfg", 8.0),
        ("denoise", 1.0),
        ("inputs", {"latent": ("ANCESTOR", 1, 0), "model": ("ANCESTOR", 2, 0)}),
        ("seed", 123456789),
    ]
    sig2 = [
        "KSampler",
        0,
        ("cfg", 8.0),
        ("denoise", 1.0),
        ("inputs", {"model": ("ANCESTOR", 2, 0), "latent": ("ANCESTOR", 1, 0)}),
        ("seed", 123456789),
    ]

    h1 = to_hashable(sig1)
    h2 = to_hashable(sig2)

    assert h1 == h2
    assert hash(h1) == hash(h2)
