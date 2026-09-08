import enum
import unittest
from comfy_execution.caching import to_hashable, Unhashable

class ColorEnum(enum.Enum):
    RED = 1
    BLUE = 2

class CustomObject:
    pass

class TestCachingToHashable(unittest.TestCase):
    def test_primitive_types(self):
        self.assertEqual(to_hashable(123), 123)
        self.assertEqual(to_hashable(45.67), 45.67)
        self.assertEqual(to_hashable("hello"), "hello")
        self.assertEqual(to_hashable(True), True)
        self.assertEqual(to_hashable(b"bytes"), b"bytes")
        self.assertIsNone(to_hashable(None))

    def test_enum_support(self):
        self.assertEqual(to_hashable(ColorEnum.RED), ColorEnum.RED)

    def test_sequence_types(self):
        lst = [1, "test", [2, 3]]
        tpl = (1, "test", (2, 3))
        res_lst = to_hashable(lst)
        res_tpl = to_hashable(tpl)

        self.assertIsInstance(res_lst, tuple)
        self.assertIsInstance(res_tpl, tuple)
        self.assertEqual(res_lst, (1, "test", (2, 3)))
        self.assertEqual(res_tpl, (1, "test", (2, 3)))
        self.assertEqual(hash(res_lst), hash(res_tpl))

    def test_mapping_types(self):
        data = {"b": 2, "a": [1, 2]}
        res = to_hashable(data)
        self.assertIsInstance(res, frozenset)
        expected = frozenset([("a", (1, 2)), ("b", 2)])
        self.assertEqual(res, expected)
        self.assertEqual(hash(res), hash(expected))

    def test_unhashable_object(self):
        obj = CustomObject()
        res = to_hashable(obj)
        self.assertIsInstance(res, Unhashable)

if __name__ == "__main__":
    unittest.main()
