import sys
from unittest.mock import MagicMock

# Mock modules that might not be installed in lightweight test environments
if "psutil" not in sys.modules:
    sys.modules["psutil"] = MagicMock()
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
if "nodes" not in sys.modules:
    sys.modules["nodes"] = MagicMock()

import unittest
from comfy_execution.caching import to_hashable, Unhashable


class TestToHashable(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(to_hashable(42), 42)
        self.assertEqual(to_hashable(3.14), 3.14)
        self.assertEqual(to_hashable("test"), "test")
        self.assertEqual(to_hashable(True), True)
        self.assertIsNone(to_hashable(None))
        self.assertEqual(to_hashable(b"bytes"), b"bytes")

    def test_sequences(self):
        res_list = to_hashable([1, 2, "a"])
        self.assertIsInstance(res_list, tuple)
        self.assertEqual(res_list, (1, 2, "a"))

        res_tuple = to_hashable((1, 2, "a"))
        self.assertIsInstance(res_tuple, tuple)
        self.assertEqual(res_tuple, (1, 2, "a"))

    def test_mappings(self):
        res_dict = to_hashable({"b": 2, "a": 1})
        self.assertIsInstance(res_dict, frozenset)
        self.assertEqual(res_dict, frozenset([("a", 1), ("b", 2)]))

    def test_sets(self):
        res_set = to_hashable({1, 2, 3})
        self.assertIsInstance(res_set, frozenset)
        self.assertEqual(res_set, frozenset([1, 2, 3]))

        res_frozenset = to_hashable(frozenset([1, 2, 3]))
        self.assertIsInstance(res_frozenset, frozenset)
        self.assertEqual(res_frozenset, frozenset([1, 2, 3]))

    def test_nested_structures(self):
        input_data = [
            "KSampler",
            None,
            [("seed", 42), ("steps", 20), ("cfg", 7.5)],
            {"model": ["ANCESTOR", 0, 0]},
        ]
        res = to_hashable(input_data)
        # Check hashability
        h = hash(res)
        self.assertIsInstance(h, int)

        # Structural equality check
        res2 = to_hashable(input_data)
        self.assertEqual(res, res2)
        self.assertEqual(hash(res), hash(res2))

    def test_unhashable(self):
        class CustomObj:
            pass

        res = to_hashable(CustomObj())
        self.assertIsInstance(res, Unhashable)


if __name__ == "__main__":
    unittest.main()
