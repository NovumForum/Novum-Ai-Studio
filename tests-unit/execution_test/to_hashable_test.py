from comfy_execution.caching import to_hashable, Unhashable


def test_to_hashable_primitives():
    assert to_hashable(42) == 42
    assert to_hashable(3.14) == 3.14
    assert to_hashable("hello") == "hello"
    assert to_hashable(True) is True
    assert to_hashable(b"data") == b"data"
    assert to_hashable(None) is None


def test_to_hashable_dicts():
    d = {"b": 2, "a": 1}
    res = to_hashable(d)
    assert isinstance(res, frozenset)
    assert res == frozenset([("a", 1), ("b", 2)])


def test_to_hashable_lists_and_tuples():
    lst = [10, "item", True]
    res_lst = to_hashable(lst)
    assert isinstance(res_lst, frozenset)
    assert res_lst == frozenset([(0, 10), (1, "item"), (2, True)])

    tup = (10, "item", True)
    res_tup = to_hashable(tup)
    assert isinstance(res_tup, frozenset)
    assert res_tup == res_lst


def test_to_hashable_nested_structures():
    nested = {
        "node_id": "1",
        "inputs": {"seed": 123, "tags": ["a", "b"]},
        "flags": (True, False),
    }
    res = to_hashable(nested)
    assert isinstance(res, frozenset)
    assert hash(res) == hash(res)


def test_to_hashable_unhashable():
    class DummyObj:
        pass

    res = to_hashable(DummyObj())
    assert isinstance(res, Unhashable)
