import pytest
from comfy_extras.nodes_color import ColorToRGBInt
from comfy_api.latest import io

class TestColorToRGBIntTooltip:
    def test_schema_properties(self):
        """Verify that ColorToRGBInt schema contains search aliases, descriptions, and correct tooltips."""
        schema = ColorToRGBInt.define_schema()
        assert schema.node_id == "ColorToRGBInt"
        assert schema.display_name == "Color to RGB Int"
        assert schema.category == "utils"
        assert schema.description == "Convert a color to a RGB integer value."

        # Verify search aliases
        assert schema.search_aliases == [
            "hex to rgb",
            "color code converter",
            "rgb integer",
            "hex color",
            "color converter"
        ]

        # Verify inputs and tooltips
        assert len(schema.inputs) == 1
        color_input = schema.inputs[0]
        assert color_input.id == "color"
        assert color_input.tooltip == "The color in HEX format (e.g. #FF0000 for Red) to convert to an integer."

        # Verify outputs and tooltips
        assert len(schema.outputs) == 1
        rgb_int_output = schema.outputs[0]
        assert rgb_int_output.display_name == "rgb_int"
        assert rgb_int_output.tooltip == "The RGB color value represented as a 24-bit integer."

    def test_execution_valid_colors(self):
        """Verify correct integer RGB conversion for valid colors."""
        # Red: FF0000 -> 255 * 256 * 256 + 0 + 0 = 16711680
        out_red = ColorToRGBInt.execute(color="#FF0000")
        assert out_red.result == (16711680,)

        # Green: 00FF00 -> 0 + 255 * 256 + 0 = 65280
        out_green = ColorToRGBInt.execute(color="#00FF00")
        assert out_green.result == (65280,)

        # Blue: 0000FF -> 255 = 255
        out_blue = ColorToRGBInt.execute(color="#0000FF")
        assert out_blue.result == (255,)

        # White: FFFFFF -> 16777215
        out_white = ColorToRGBInt.execute(color="#FFFFFF")
        assert out_white.result == (16777215,)

        # Black: 000000 -> 0
        out_black = ColorToRGBInt.execute(color="#000000")
        assert out_black.result == (0,)

    def test_execution_invalid_colors(self):
        """Verify that malformed color values raise a ValueError."""
        invalid_values = [
            "#FFF",
            "FF0000",
            "#FF00000",
            "",
            "RGB(255,0,0)",
            "#GG0000"  # Invalid hex characters (will raise ValueError during int conversion or formatting)
        ]
        for val in invalid_values:
            with pytest.raises(ValueError):
                ColorToRGBInt.execute(color=val)
