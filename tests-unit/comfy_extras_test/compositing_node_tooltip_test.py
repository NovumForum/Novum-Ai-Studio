import ast
import unittest


class TestCompositingNodeTooltips(unittest.TestCase):
    def setUp(self):
        with open("comfy_extras/nodes_compositing.py", "r", encoding="utf-8") as f:
            self.tree = ast.parse(f.read())

    def get_class_node(self, class_name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def test_porter_duff_composite_schema(self):
        class_node = self.get_class_node("PorterDuffImageComposite")
        self.assertIsNotNone(class_node)
        source = ast.unparse(class_node)

        self.assertIn(
            "description='Composites two images with alpha channels using Porter-Duff blending operations.'",
            source,
        )
        self.assertIn("tooltip='Source foreground image to composite.'", source)
        self.assertIn("tooltip='Alpha transparency mask for the source image.'", source)
        self.assertIn(
            "tooltip='Destination background image to composite onto.'", source
        )
        self.assertIn(
            "tooltip='Alpha transparency mask for the destination image.'", source
        )
        self.assertIn(
            "tooltip='Porter-Duff compositing operator mode (e.g. DST_OVER, SRC_OVER).'",
            source,
        )
        self.assertIn("tooltip='Composited result image.'", source)
        self.assertIn("tooltip='Combined alpha mask result.'", source)

    def test_split_image_with_alpha_schema(self):
        class_node = self.get_class_node("SplitImageWithAlpha")
        self.assertIsNotNone(class_node)
        source = ast.unparse(class_node)

        self.assertIn(
            "description='Splits an RGBA image into an RGB image tensor and an inverted alpha mask tensor.'",
            source,
        )
        self.assertIn(
            "tooltip='Input image tensor containing RGB or RGBA channels.'", source
        )
        self.assertIn("tooltip='RGB image without alpha channel.'", source)
        self.assertIn("tooltip='Extracted alpha mask (inverted).'", source)
        self.assertIn("'split rgba'", source)

    def test_join_image_with_alpha_schema(self):
        class_node = self.get_class_node("JoinImageWithAlpha")
        self.assertIsNotNone(class_node)
        source = ast.unparse(class_node)

        self.assertIn(
            "description='Combines an RGB image tensor with a mask tensor into a single RGBA image with transparency.'",
            source,
        )
        self.assertIn("tooltip='RGB image tensor to attach transparency to.'", source)
        self.assertIn(
            "tooltip='Mask tensor to use as the alpha transparency channel.'", source
        )
        self.assertIn("tooltip='Combined RGBA image with transparency.'", source)
        self.assertIn("'combine alpha'", source)


if __name__ == "__main__":
    unittest.main()
