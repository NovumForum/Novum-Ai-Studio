import ast
import unittest


class TestCLIPSDXLNodeTooltips(unittest.TestCase):
    def setUp(self):
        with open("comfy_extras/nodes_clip_sdxl.py", "r", encoding="utf-8") as f:
            self.tree = ast.parse(f.read())

    def _get_class_node(self, class_name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _get_schema_call(self, class_node):
        for stmt in ast.walk(class_node):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute) and stmt.func.attr == "Schema":
                    return stmt
        return None

    def test_clip_text_encode_sdxl_refiner_schema(self):
        class_node = self._get_class_node("CLIPTextEncodeSDXLRefiner")
        self.assertIsNotNone(class_node, "CLIPTextEncodeSDXLRefiner class not found")

        schema_call = self._get_schema_call(class_node)
        self.assertIsNotNone(schema_call, "Schema call not found in CLIPTextEncodeSDXLRefiner")

        keywords = {kw.arg: kw.value for kw in schema_call.keywords}
        self.assertIn("display_name", keywords)
        self.assertEqual(ast.literal_eval(keywords["display_name"]), "CLIP Text Encode (SDXL Refiner)")

        self.assertIn("description", keywords)
        self.assertTrue(len(ast.literal_eval(keywords["description"])) > 0)

        self.assertIn("search_aliases", keywords)
        aliases = ast.literal_eval(keywords["search_aliases"])
        self.assertIn("sdxl refiner prompt", aliases)

        # Check input tooltips
        inputs_list = keywords.get("inputs")
        self.assertIsNotNone(inputs_list)
        inputs_tooltips = {}
        for elt in inputs_list.elts:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "Input":
                param_name = ast.literal_eval(elt.args[0])
                kw_dict = {kw.arg: kw.value for kw in elt.keywords}
                self.assertIn("tooltip", kw_dict, f"Missing tooltip for input '{param_name}' in CLIPTextEncodeSDXLRefiner")
                inputs_tooltips[param_name] = ast.literal_eval(kw_dict["tooltip"])

        expected_inputs = ["ascore", "width", "height", "text", "clip"]
        for inp in expected_inputs:
            self.assertIn(inp, inputs_tooltips)
            self.assertTrue(len(inputs_tooltips[inp]) > 0)

        # Check output tooltips
        outputs_list = keywords.get("outputs")
        self.assertIsNotNone(outputs_list)
        for elt in outputs_list.elts:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "Output":
                kw_dict = {kw.arg: kw.value for kw in elt.keywords}
                self.assertIn("tooltip", kw_dict, "Missing tooltip for output in CLIPTextEncodeSDXLRefiner")
                self.assertTrue(len(ast.literal_eval(kw_dict["tooltip"])) > 0)

    def test_clip_text_encode_sdxl_schema(self):
        class_node = self._get_class_node("CLIPTextEncodeSDXL")
        self.assertIsNotNone(class_node, "CLIPTextEncodeSDXL class not found")

        schema_call = self._get_schema_call(class_node)
        self.assertIsNotNone(schema_call, "Schema call not found in CLIPTextEncodeSDXL")

        keywords = {kw.arg: kw.value for kw in schema_call.keywords}
        self.assertIn("display_name", keywords)
        self.assertEqual(ast.literal_eval(keywords["display_name"]), "CLIP Text Encode (SDXL)")

        self.assertIn("description", keywords)
        self.assertTrue(len(ast.literal_eval(keywords["description"])) > 0)

        self.assertIn("search_aliases", keywords)
        aliases = ast.literal_eval(keywords["search_aliases"])
        self.assertIn("sdxl prompt", aliases)

        # Check input tooltips
        inputs_list = keywords.get("inputs")
        self.assertIsNotNone(inputs_list)
        inputs_tooltips = {}
        for elt in inputs_list.elts:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "Input":
                param_name = ast.literal_eval(elt.args[0])
                kw_dict = {kw.arg: kw.value for kw in elt.keywords}
                self.assertIn("tooltip", kw_dict, f"Missing tooltip for input '{param_name}' in CLIPTextEncodeSDXL")
                inputs_tooltips[param_name] = ast.literal_eval(kw_dict["tooltip"])

        expected_inputs = ["clip", "width", "height", "crop_w", "crop_h", "target_width", "target_height", "text_g", "text_l"]
        for inp in expected_inputs:
            self.assertIn(inp, inputs_tooltips)
            self.assertTrue(len(inputs_tooltips[inp]) > 0)

        # Check output tooltips
        outputs_list = keywords.get("outputs")
        self.assertIsNotNone(outputs_list)
        for elt in outputs_list.elts:
            if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "Output":
                kw_dict = {kw.arg: kw.value for kw in elt.keywords}
                self.assertIn("tooltip", kw_dict, "Missing tooltip for output in CLIPTextEncodeSDXL")
                self.assertTrue(len(ast.literal_eval(kw_dict["tooltip"])) > 0)


if __name__ == "__main__":
    unittest.main()
