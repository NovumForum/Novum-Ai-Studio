import ast
import os
import unittest


class TestLatentNodeTooltips(unittest.TestCase):
    def setUp(self):
        self.nodes_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "comfy_extras", "nodes_latent.py"
        )
        with open(self.nodes_path, "r", encoding="utf-8") as f:
            self.tree = ast.parse(f.read())

    def _get_class_node(self, class_name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _get_schema_keywords(self, class_node):
        for stmt in class_node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "define_schema":
                for sub_node in ast.walk(stmt):
                    if isinstance(sub_node, ast.Call):
                        # check if calling io.Schema
                        if (
                            isinstance(sub_node.func, ast.Attribute)
                            and sub_node.func.attr == "Schema"
                        ):
                            return {kw.arg: kw.value for kw in sub_node.keywords}
        return {}

    def test_latent_add_schema_metadata(self):
        class_node = self._get_class_node("LatentAdd")
        self.assertIsNotNone(class_node)
        kwargs = self._get_schema_keywords(class_node)
        self.assertIn("description", kwargs)
        self.assertIsInstance(kwargs["description"], ast.Constant)
        self.assertTrue(len(kwargs["description"].value) > 0)

        # Check inputs and outputs tooltips
        inputs = kwargs.get("inputs")
        self.assertIsInstance(inputs, ast.List)
        for elt in inputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

        outputs = kwargs.get("outputs")
        self.assertIsInstance(outputs, ast.List)
        for elt in outputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

    def test_latent_subtract_schema_metadata(self):
        class_node = self._get_class_node("LatentSubtract")
        self.assertIsNotNone(class_node)
        kwargs = self._get_schema_keywords(class_node)
        self.assertIn("description", kwargs)
        self.assertIsInstance(kwargs["description"], ast.Constant)

        inputs = kwargs.get("inputs")
        self.assertIsInstance(inputs, ast.List)
        for elt in inputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

        outputs = kwargs.get("outputs")
        self.assertIsInstance(outputs, ast.List)
        for elt in outputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

    def test_latent_multiply_schema_metadata(self):
        class_node = self._get_class_node("LatentMultiply")
        self.assertIsNotNone(class_node)
        kwargs = self._get_schema_keywords(class_node)
        self.assertIn("description", kwargs)
        self.assertIsInstance(kwargs["description"], ast.Constant)

        inputs = kwargs.get("inputs")
        self.assertIsInstance(inputs, ast.List)
        for elt in inputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

        outputs = kwargs.get("outputs")
        self.assertIsInstance(outputs, ast.List)
        for elt in outputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

    def test_latent_interpolate_schema_metadata(self):
        class_node = self._get_class_node("LatentInterpolate")
        self.assertIsNotNone(class_node)
        kwargs = self._get_schema_keywords(class_node)
        self.assertIn("description", kwargs)
        self.assertIsInstance(kwargs["description"], ast.Constant)

        inputs = kwargs.get("inputs")
        self.assertIsInstance(inputs, ast.List)
        for elt in inputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

        outputs = kwargs.get("outputs")
        self.assertIsInstance(outputs, ast.List)
        for elt in outputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

    def test_latent_concat_schema_metadata(self):
        class_node = self._get_class_node("LatentConcat")
        self.assertIsNotNone(class_node)
        kwargs = self._get_schema_keywords(class_node)
        self.assertIn("description", kwargs)
        self.assertIsInstance(kwargs["description"], ast.Constant)

        inputs = kwargs.get("inputs")
        self.assertIsInstance(inputs, ast.List)
        for elt in inputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)

        outputs = kwargs.get("outputs")
        self.assertIsInstance(outputs, ast.List)
        for elt in outputs.elts:
            kw_dict = {kw.arg: kw.value for kw in elt.keywords}
            self.assertIn("tooltip", kw_dict)


if __name__ == "__main__":
    unittest.main()
