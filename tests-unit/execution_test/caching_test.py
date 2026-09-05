import sys
import unittest
from unittest.mock import MagicMock

# Mock dependencies not installed in sandbox environment for pure logic unit testing
sys.modules['psutil'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['torch'] = MagicMock()

from comfy_execution.caching import to_hashable, Unhashable


class TestToHashable(unittest.TestCase):
    def test_primitive_types(self):
        self.assertEqual(to_hashable(123), 123)
        self.assertEqual(to_hashable(3.14), 3.14)
        self.assertEqual(to_hashable("hello"), "hello")
        self.assertEqual(to_hashable(True), True)
        self.assertEqual(to_hashable(b"bytes"), b"bytes")
        self.assertIsNone(to_hashable(None))

    def test_sequence_types(self):
        obj = [1, "test", [2, 3]]
        expected = (1, "test", (2, 3))
        res = to_hashable(obj)
        self.assertEqual(res, expected)
        self.assertTrue(isinstance(hash(res), int))

    def test_mapping_types(self):
        obj = {"b": 2, "a": [1, 2]}
        res = to_hashable(obj)
        expected = (("a", (1, 2)), ("b", 2))
        self.assertEqual(res, expected)
        self.assertTrue(isinstance(hash(res), int))

    def test_nested_structures(self):
        obj = {
            "node_1": ["KSampler", {"seed": 42, "cfg": 8.0}],
            "node_2": [("ANCESTOR", 0, "LATENT")],
        }
        res = to_hashable(obj)
        self.assertTrue(isinstance(hash(res), int))

    def test_unhashable_objects(self):
        class CustomObj:
            pass

        res = to_hashable(CustomObj())
        self.assertTrue(isinstance(res, Unhashable))


if __name__ == "__main__":
    unittest.main()
