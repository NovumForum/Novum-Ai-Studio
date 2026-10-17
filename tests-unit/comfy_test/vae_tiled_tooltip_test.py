import ast
import os
import unittest


class TestVAETiledTooltips(unittest.TestCase):
    def setUp(self):
        self.nodes_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "nodes.py"
        )
        with open(self.nodes_file, "r") as f:
            self.tree = ast.parse(f.read(), self.nodes_file)

    def get_class_node(self, class_name):
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def test_vae_decode_tiled_schema(self):
        class_node = self.get_class_node("VAEDecodeTiled")
        self.assertIsNotNone(class_node, "VAEDecodeTiled class not found")

        # Check DESCRIPTION, SEARCH_ALIASES, OUTPUT_TOOLTIPS
        has_desc = False
        has_aliases = False
        has_output_tooltips = False

        for stmt in class_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "DESCRIPTION":
                            has_desc = True
                        elif target.id == "SEARCH_ALIASES":
                            has_aliases = True
                        elif target.id == "OUTPUT_TOOLTIPS":
                            has_output_tooltips = True

        self.assertTrue(has_desc, "VAEDecodeTiled should have DESCRIPTION")
        self.assertTrue(has_aliases, "VAEDecodeTiled should have SEARCH_ALIASES")
        self.assertTrue(
            has_output_tooltips, "VAEDecodeTiled should have OUTPUT_TOOLTIPS"
        )

    def test_vae_encode_tiled_schema(self):
        class_node = self.get_class_node("VAEEncodeTiled")
        self.assertIsNotNone(class_node, "VAEEncodeTiled class not found")

        # Check DESCRIPTION, SEARCH_ALIASES, OUTPUT_TOOLTIPS
        has_desc = False
        has_aliases = False
        has_output_tooltips = False

        for stmt in class_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "DESCRIPTION":
                            has_desc = True
                        elif target.id == "SEARCH_ALIASES":
                            has_aliases = True
                        elif target.id == "OUTPUT_TOOLTIPS":
                            has_output_tooltips = True

        self.assertTrue(has_desc, "VAEEncodeTiled should have DESCRIPTION")
        self.assertTrue(has_aliases, "VAEEncodeTiled should have SEARCH_ALIASES")
        self.assertTrue(
            has_output_tooltips, "VAEEncodeTiled should have OUTPUT_TOOLTIPS"
        )


if __name__ == "__main__":
    unittest.main()
