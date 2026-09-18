import sys
from types import ModuleType


# Utility to dynamically create mock modules for any uninstalled dependencies
class DummyModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        submod = DummyModule(f"{self.__name__}.{name}")
        setattr(self, name, submod)
        return submod

    def __call__(self, *args, **kwargs):
        return DummyModule("call_result")


def ensure_mock_module(mod_name):
    parts = mod_name.split('.')
    for i in range(1, len(parts) + 1):
        target = '.'.join(parts[:i])
        if target not in sys.modules:
            sys.modules[target] = DummyModule(target)


for mod in [
    'torch',
    'av',
    'av.container',
    'av.subtitles',
    'av.subtitles.stream',
    'PIL',
    'PIL.PngImagePlugin',
    'PIL.Image',
    'scipy',
    'numpy',
    'torchvision',
    'torchaudio',
    'tqdm',
]:
    ensure_mock_module(mod)

import unittest
from comfy_extras.nodes_string import (
    StringConcatenate,
    StringSubstring,
    StringLength,
    CaseConverter,
    StringTrim,
    StringReplace,
    StringContains,
    StringCompare,
    RegexMatch,
    RegexExtract,
    RegexReplace,
)


class TestStringNodeTooltips(unittest.TestCase):
    def test_string_node_schemas(self):
        nodes = [
            StringConcatenate,
            StringSubstring,
            StringLength,
            CaseConverter,
            StringTrim,
            StringReplace,
            StringContains,
            StringCompare,
            RegexMatch,
            RegexExtract,
            RegexReplace,
        ]

        for node in nodes:
            schema = node.define_schema()
            self.assertIsNotNone(schema.description, f"{node.__name__} missing description")
            self.assertTrue(len(schema.description) > 0, f"{node.__name__} description is empty")

            for inp in schema.inputs:
                self.assertIsNotNone(inp.tooltip, f"{node.__name__} input missing tooltip")
                self.assertTrue(len(inp.tooltip) > 0, f"{node.__name__} input tooltip is empty")

            for out in schema.outputs:
                self.assertIsNotNone(out.tooltip, f"{node.__name__} output missing tooltip")
                self.assertTrue(len(out.tooltip) > 0, f"{node.__name__} output tooltip is empty")

    def test_string_node_execution(self):
        # Concatenate
        res = StringConcatenate.execute("hello", "world", " ")
        self.assertEqual(res.args[0], "hello world")

        # Substring
        res = StringSubstring.execute("hello world", 0, 5)
        self.assertEqual(res.args[0], "hello")

        # Length
        res = StringLength.execute("hello")
        self.assertEqual(res.args[0], 5)

        # CaseConverter
        res = CaseConverter.execute("hello world", "UPPERCASE")
        self.assertEqual(res.args[0], "HELLO WORLD")

        # Trim
        res = StringTrim.execute("  hello  ", "Both")
        self.assertEqual(res.args[0], "hello")

        # Replace
        res = StringReplace.execute("hello world", "world", "ComfyUI")
        self.assertEqual(res.args[0], "hello ComfyUI")

        # Contains
        res = StringContains.execute("hello world", "world", True)
        self.assertTrue(res.args[0])

        # Compare
        res = StringCompare.execute("hello world", "hello", "Starts With", True)
        self.assertTrue(res.args[0])

        # RegexMatch
        res = RegexMatch.execute("hello world 123", r"\d+", True, False, False)
        self.assertTrue(res.args[0])

        # RegexExtract
        res = RegexExtract.execute("hello 123 world", r"\d+", "First Match", True, False, False, 1)
        self.assertEqual(res.args[0], "123")

        # RegexReplace
        res = RegexReplace.execute("hello 123 world", r"\d+", "456")
        self.assertEqual(res.args[0], "hello 456 world")


if __name__ == "__main__":
    unittest.main()
