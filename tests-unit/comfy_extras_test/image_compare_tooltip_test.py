import ast
import os
import unittest


class ImageCompareTooltipTest(unittest.TestCase):
    def test_image_compare_schema_tooltips_and_aliases(self):
        filepath = os.path.join("comfy_extras", "nodes_image_compare.py")
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        node_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ImageCompare":
                node_class = node
                break

        self.assertIsNotNone(node_class, "ImageCompare class not found")

        schema_call = None
        for node in ast.walk(node_class):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Schema"
            ):
                schema_call = node
                break

        self.assertIsNotNone(schema_call, "IO.Schema call not found in ImageCompare")

        # Verify search_aliases kwarg
        search_aliases_kw = None
        inputs_kw = None
        for kw in schema_call.keywords:
            if kw.arg == "search_aliases":
                search_aliases_kw = kw
            elif kw.arg == "inputs":
                inputs_kw = kw

        self.assertIsNotNone(search_aliases_kw, "search_aliases not defined in ImageCompare schema")
        self.assertIsInstance(search_aliases_kw.value, ast.List, "search_aliases must be a list")
        aliases = [elt.value if isinstance(elt, ast.Constant) else elt.s for elt in search_aliases_kw.value.elts]
        self.assertIn("image comparison", aliases)
        self.assertIn("side by side", aliases)

        # Verify input tooltips
        self.assertIsNotNone(inputs_kw, "inputs keyword argument not found in schema")
        self.assertIsInstance(inputs_kw.value, ast.List, "inputs must be a list")

        expected_tooltips = {
            "image_a": "First image or image batch to compare.",
            "image_b": "Second image or image batch to compare.",
            "compare_view": "Interactive side-by-side visual comparison view widget.",
        }

        found_tooltips = {}
        for input_call in inputs_kw.value.elts:
            if isinstance(input_call, ast.Call):
                input_name = None
                tooltip_val = None
                if input_call.args:
                    arg0 = input_call.args[0]
                    if isinstance(arg0, ast.Constant):
                        input_name = arg0.value

                for kw in input_call.keywords:
                    if kw.arg == "tooltip" and isinstance(kw.value, ast.Constant):
                        tooltip_val = kw.value.value

                if input_name:
                    found_tooltips[input_name] = tooltip_val

        for param_name, expected_tooltip in expected_tooltips.items():
            self.assertIn(param_name, found_tooltips, f"Missing tooltip for parameter {param_name}")
            self.assertEqual(
                found_tooltips[param_name],
                expected_tooltip,
                f"Mismatch in tooltip for parameter {param_name}",
            )


if __name__ == "__main__":
    unittest.main()
