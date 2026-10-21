import ast
import unittest


class TestPostProcessingNodeTooltips(unittest.TestCase):
    def test_post_processing_node_schemas(self):
        with open("comfy_extras/nodes_post_processing.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        nodes_to_check = {
            "Blend": {
                "display_name": "Image Blend",
                "description": "Blend two images together using specified blend modes and opacity factors.",
                "search_aliases": ["blend", "image blend", "mix images", "combine images", "overlay images", "composite"],
                "inputs": ["image1", "image2", "blend_factor", "blend_mode"],
                "outputs": True,
            },
            "Blur": {
                "display_name": "Image Blur",
                "description": "Apply a gaussian blur to an image with customizable radius and standard deviation.",
                "search_aliases": ["blur", "image blur", "gaussian blur", "smooth image", "soften"],
                "inputs": ["image", "blur_radius", "sigma"],
                "outputs": True,
            },
            "Quantize": {
                "display_name": "Image Quantize",
                "description": "Reduce the color palette of an image with optional dithering algorithms.",
                "search_aliases": ["quantize", "color palette", "dither", "posterize", "pixel art", "reduce colors"],
                "inputs": ["image", "colors", "dither"],
                "outputs": True,
            },
            "Sharpen": {
                "display_name": "Image Sharpen",
                "description": "Sharpen an image by enhancing edges using an unsharp masking kernel.",
                "search_aliases": ["sharpen", "image sharpen", "unsharp mask", "enhance edges", "clarity"],
                "inputs": ["image", "sharpen_radius", "sigma", "alpha"],
                "outputs": True,
            },
            "ImageScaleToTotalPixels": {
                "display_name": "Image Scale to Total Pixels",
                "description": "Scale an image to a target megapixel count while preserving its aspect ratio.",
                "search_aliases": ["scale to megapixels", "megapixels", "resize megapixels", "image scale total pixels", "pixel count"],
                "inputs": ["image", "upscale_method", "megapixels", "resolution_steps"],
                "outputs": True,
            },
        }

        found_nodes = set()

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name in nodes_to_check:
                found_nodes.add(stmt.name)
                expected = nodes_to_check[stmt.name]

                # Find define_schema method
                schema_method = None
                for item in stmt.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                        schema_method = item
                        break

                self.assertIsNotNone(schema_method, f"define_schema not found in {stmt.name}")

                # Locate return io.Schema(...)
                schema_call = None
                for node in ast.walk(schema_method):
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                        schema_call = node.value
                        break

                self.assertIsNotNone(schema_call, f"Schema return call not found in {stmt.name}")

                # Extract keyword arguments passed to io.Schema
                kwargs = {kw.arg: kw.value for kw in schema_call.keywords}

                # Verify display_name
                self.assertIn("display_name", kwargs, f"display_name missing in {stmt.name}")
                self.assertEqual(kwargs["display_name"].value, expected["display_name"])

                # Verify description
                self.assertIn("description", kwargs, f"description missing in {stmt.name}")
                self.assertEqual(kwargs["description"].value, expected["description"])

                # Verify search_aliases
                self.assertIn("search_aliases", kwargs, f"search_aliases missing in {stmt.name}")
                aliases = [elt.value for elt in kwargs["search_aliases"].elts]
                self.assertEqual(aliases, expected["search_aliases"])

                # Verify input tooltips
                self.assertIn("inputs", kwargs, f"inputs missing in {stmt.name}")
                inputs_list = kwargs["inputs"]
                found_inputs = []

                for input_elt in inputs_list.elts:
                    if isinstance(input_elt, ast.Call):
                        input_kwargs = {kw.arg: kw.value for kw in input_elt.keywords}
                        input_name = input_elt.args[0].value if input_elt.args else None
                        if input_name:
                            found_inputs.append(input_name)
                            self.assertIn(
                                "tooltip",
                                input_kwargs,
                                f"Input '{input_name}' missing tooltip in {stmt.name}",
                            )
                            self.assertTrue(
                                len(input_kwargs["tooltip"].value) > 0,
                                f"Input '{input_name}' tooltip is empty in {stmt.name}",
                            )

                self.assertEqual(
                    found_inputs,
                    expected["inputs"],
                    f"Inputs mismatch in {stmt.name}",
                )

                # Verify output tooltips
                self.assertIn("outputs", kwargs, f"outputs missing in {stmt.name}")
                outputs_list = kwargs["outputs"]
                for output_elt in outputs_list.elts:
                    if isinstance(output_elt, ast.Call):
                        output_kwargs = {kw.arg: kw.value for kw in output_elt.keywords}
                        self.assertIn(
                            "tooltip",
                            output_kwargs,
                            f"Output missing tooltip in {stmt.name}",
                        )
                        self.assertTrue(
                            len(output_kwargs["tooltip"].value) > 0,
                            f"Output tooltip is empty in {stmt.name}",
                        )

        self.assertEqual(found_nodes, set(nodes_to_check.keys()), "Not all nodes were checked")


if __name__ == "__main__":
    unittest.main()
