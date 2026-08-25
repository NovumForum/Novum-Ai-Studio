import sys
from unittest.mock import MagicMock
import pytest

# Mock dependencies not available in clean environment
sys.modules['torch'] = MagicMock()
mock_av = MagicMock()
sys.modules['av'] = mock_av
sys.modules['av.container'] = mock_av
sys.modules['av.subtitles'] = mock_av
sys.modules['av.subtitles.stream'] = mock_av
sys.modules['numpy'] = MagicMock()
mock_pil = MagicMock()
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil
sys.modules['PIL.PngImagePlugin'] = mock_pil
sys.modules['tqdm'] = MagicMock()

from comfy_extras.nodes_color import ColorToRGBInt


def test_color_to_rgb_int_schema():
    schema = ColorToRGBInt.GET_SCHEMA()

    assert schema.node_id == "ColorToRGBInt"
    assert schema.display_name == "Color to RGB Int"
    assert schema.category == "utils"
    assert schema.description == "Convert a color to a RGB integer value."
    assert schema.search_aliases == ["hex to rgb", "color code", "color converter", "rgb int", "hex to int"]

    # Inputs check
    assert len(schema.inputs) == 1
    color_input = schema.inputs[0]
    assert color_input.id == "color"
    assert color_input.tooltip == "The hex color code (e.g. #RRGGBB) to convert into an integer RGB value."

    # Outputs check
    assert len(schema.outputs) == 1
    rgb_output = schema.outputs[0]
    assert rgb_output.id == "rgb_int"
    assert rgb_output.display_name == "rgb_int"
    assert rgb_output.tooltip == "The converted integer RGB value representing the color."


def test_color_to_rgb_int_execution():
    # Test valid hex values
    res_red = ColorToRGBInt.execute("#FF0000")
    assert res_red.args == (16711680,)

    res_green = ColorToRGBInt.execute("#00FF00")
    assert res_green.args == (65280,)

    res_blue = ColorToRGBInt.execute("#0000FF")
    assert res_blue.args == (255,)

    res_black = ColorToRGBInt.execute("#000000")
    assert res_black.args == (0,)

    res_white = ColorToRGBInt.execute("#FFFFFF")
    assert res_white.args == (16777215,)

    # Test invalid color format raises ValueError
    with pytest.raises(ValueError, match="Color must be in format #RRGGBB"):
        ColorToRGBInt.execute("FF0000")

    with pytest.raises(ValueError, match="Color must be in format #RRGGBB"):
        ColorToRGBInt.execute("#FF00")
