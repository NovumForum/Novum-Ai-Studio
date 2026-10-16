import ast
import os
import unittest


class TestHyperTileFreeUTooltips(unittest.TestCase):
    def _parse_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=filepath)

    def test_hypertile_schema(self):
        filepath = os.path.join("comfy_extras", "nodes_hypertile.py")
        tree = self._parse_file(filepath)

        found_hypertile = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "HyperTile":
                found_hypertile = True
                # Check for define_schema method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                        schema_code = ast.unparse(item)
                        self.assertIn("description=", schema_code)
                        self.assertIn("search_aliases=", schema_code)
                        self.assertIn("tooltip=", schema_code)

        self.assertTrue(found_hypertile, "HyperTile class not found")

    def test_freelunch_schemas(self):
        filepath = os.path.join("comfy_extras", "nodes_freelunch.py")
        tree = self._parse_file(filepath)

        found_freeu = False
        found_freeu_v2 = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in ("FreeU", "FreeU_V2"):
                    if node.name == "FreeU":
                        found_freeu = True
                    if node.name == "FreeU_V2":
                        found_freeu_v2 = True

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                            schema_code = ast.unparse(item)
                            self.assertIn("description=", schema_code)
                            self.assertIn("search_aliases=", schema_code)
                            self.assertIn("tooltip=", schema_code)

        self.assertTrue(found_freeu, "FreeU class not found")
        self.assertTrue(found_freeu_v2, "FreeU_V2 class not found")


if __name__ == "__main__":
    unittest.main()
