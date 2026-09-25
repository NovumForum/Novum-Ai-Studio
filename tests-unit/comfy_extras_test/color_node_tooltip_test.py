import unittest
from comfy_extras.nodes_color import ColorToRGBInt


class TestColorToRGBIntSchemaAndExecution(unittest.TestCase):
    def test_color_to_rgb_int_schema(self):
        schema = ColorToRGBInt.define_schema()
        self.assertEqual(schema.node_id, "ColorToRGBInt")
        self.assertEqual(schema.display_name, "Color to RGB Int")
        self.assertEqual(schema.description, "Convert a color to a RGB integer value.")
        self.assertIn("hex to rgb", schema.search_aliases)
        self.assertIn("color code", schema.search_aliases)
        self.assertIn("rgb integer", schema.search_aliases)

        # Inputs check
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(schema.inputs[0].id, "color")
        self.assertEqual(
            schema.inputs[0].tooltip,
            "Hex color string in #RRGGBB format (e.g. #FF0000).",
        )

        # Outputs check
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].display_name, "rgb_int")
        self.assertEqual(
            schema.outputs[0].tooltip,
            "Integer representation of RGB color (R * 65536 + G * 256 + B).",
        )

    def test_color_to_rgb_int_execute_valid(self):
        # Red: #FF0000 -> 255 * 65536 = 16711680
        res = ColorToRGBInt.execute("#FF0000")
        self.assertEqual(res.args[0], 16711680)

        # Green: #00FF00 -> 255 * 256 = 65280
        res = ColorToRGBInt.execute("#00FF00")
        self.assertEqual(res.args[0], 65280)

        # Blue: #0000FF -> 255
        res = ColorToRGBInt.execute("#0000FF")
        self.assertEqual(res.args[0], 255)

        # White: #FFFFFF -> 16777215
        res = ColorToRGBInt.execute("#FFFFFF")
        self.assertEqual(res.args[0], 16777215)

    def test_color_to_rgb_int_execute_invalid(self):
        with self.assertRaises(ValueError):
            ColorToRGBInt.execute("FF0000")

        with self.assertRaises(ValueError):
            ColorToRGBInt.execute("#FFF")


if __name__ == "__main__":
    unittest.main()
