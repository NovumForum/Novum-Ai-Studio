import ast
import os
import unittest


class TestVAETiledTooltips(unittest.TestCase):
    def test_vae_tiled_node_schemas(self):
        nodes_filepath = os.path.join(
            os.path.dirname(__file__), "..", "..", "nodes.py"
        )
        with open(nodes_filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="nodes.py")

        target_classes = ["VAEDecodeTiled", "VAEEncodeTiled"]
        found_classes = {}

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name in target_classes:
                has_desc = False
                has_aliases = False
                has_output_tooltips = False
                has_input_tooltips = False

                for item in stmt.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if getattr(target, "id", None) == "DESCRIPTION":
                                has_desc = True
                            elif getattr(target, "id", None) == "SEARCH_ALIASES":
                                has_aliases = True
                            elif getattr(target, "id", None) == "OUTPUT_TOOLTIPS":
                                has_output_tooltips = True
                    elif (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "INPUT_TYPES"
                    ):
                        # Check that 'tooltip' appears in the ast representation of INPUT_TYPES
                        input_types_str = ast.unparse(item)
                        if "'tooltip'" in input_types_str or '"tooltip"' in input_types_str:
                            has_input_tooltips = True

                found_classes[stmt.name] = {
                    "has_desc": has_desc,
                    "has_aliases": has_aliases,
                    "has_output_tooltips": has_output_tooltips,
                    "has_input_tooltips": has_input_tooltips,
                }

        for cls_name in target_classes:
            self.assertIn(cls_name, found_classes)
            meta = found_classes[cls_name]
            self.assertTrue(meta["has_desc"], f"{cls_name} missing DESCRIPTION")
            self.assertTrue(
                meta["has_aliases"], f"{cls_name} missing SEARCH_ALIASES"
            )
            self.assertTrue(
                meta["has_output_tooltips"],
                f"{cls_name} missing OUTPUT_TOOLTIPS",
            )
            self.assertTrue(
                meta["has_input_tooltips"],
                f"{cls_name} missing input parameter tooltips",
            )


if __name__ == "__main__":
    unittest.main()
