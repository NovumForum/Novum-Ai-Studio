import ast
import os
import unittest


class TestMorphologyNodeTooltips(unittest.TestCase):
    def setUp(self):
        self.file_path = os.path.join("comfy_extras", "nodes_morphology.py")
        self.assertTrue(os.path.exists(self.file_path), f"File not found: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.tree = ast.parse(f.read())

    def _get_class_node(self, class_name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _get_define_schema_call(self, class_node):
        for stmt in class_node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "define_schema":
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Call) and getattr(child.func, "attr", None) == "Schema":
                        return child
        return None

    def test_morphology_schema_metadata(self):
        class_node = self._get_class_node("Morphology")
        self.assertIsNotNone(class_node, "Morphology class not found")
        schema_call = self._get_define_schema_call(class_node)
        self.assertIsNotNone(schema_call, "define_schema Call not found in Morphology")

        kwargs = {kw.arg: kw.value for kw in schema_call.keywords}
        self.assertIn("description", kwargs)
        self.assertIn("search_aliases", kwargs)
        self.assertIn("display_name", kwargs)

    def test_image_rgb_to_yuv_schema_metadata(self):
        class_node = self._get_class_node("ImageRGBToYUV")
        self.assertIsNotNone(class_node, "ImageRGBToYUV class not found")
        schema_call = self._get_define_schema_call(class_node)
        self.assertIsNotNone(schema_call, "define_schema Call not found in ImageRGBToYUV")

        kwargs = {kw.arg: kw.value for kw in schema_call.keywords}
        self.assertIn("description", kwargs)
        self.assertIn("search_aliases", kwargs)
        self.assertIn("display_name", kwargs)

    def test_image_yuv_to_rgb_schema_metadata(self):
        class_node = self._get_class_node("ImageYUVToRGB")
        self.assertIsNotNone(class_node, "ImageYUVToRGB class not found")
        schema_call = self._get_define_schema_call(class_node)
        self.assertIsNotNone(schema_call, "define_schema Call not found in ImageYUVToRGB")

        kwargs = {kw.arg: kw.value for kw in schema_call.keywords}
        self.assertIn("description", kwargs)
        self.assertIn("search_aliases", kwargs)
        self.assertIn("display_name", kwargs)


if __name__ == "__main__":
    unittest.main()
