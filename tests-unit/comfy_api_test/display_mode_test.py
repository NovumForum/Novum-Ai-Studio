import sys
from unittest.mock import MagicMock

# Set up mock modules before any real imports to avoid import failures in restricted environments
torch_mock = MagicMock()
torch_mock.version.cuda = "12.1"
sys.modules.setdefault("torch", torch_mock)
sys.modules.setdefault("packaging", MagicMock())
sys.modules.setdefault("packaging.version", MagicMock())
sys.modules.setdefault("av", MagicMock())
sys.modules.setdefault("av.container", MagicMock())
sys.modules.setdefault("av.video", MagicMock())
sys.modules.setdefault("av.subtitles", MagicMock())
sys.modules.setdefault("av.subtitles.stream", MagicMock())
sys.modules.setdefault("numpy", MagicMock())
sys.modules.setdefault("numpy.dtypes", MagicMock())
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())
sys.modules.setdefault("PIL.PngImagePlugin", MagicMock())
sys.modules.setdefault("tqdm", MagicMock())
sys.modules.setdefault("spandrel", MagicMock())
sys.modules.setdefault("comfy.utils", MagicMock())
sys.modules.setdefault("comfy.cli_args", MagicMock())
sys.modules.setdefault("folder_paths", MagicMock())
sys.modules.setdefault("scipy", MagicMock())
sys.modules.setdefault("scipy.ndimage", MagicMock())
sys.modules.setdefault("node_helpers", MagicMock())

nodes_mock = MagicMock()
nodes_mock.MAX_RESOLUTION = 16384
sys.modules.setdefault("nodes", nodes_mock)

from comfy_api.latest._io import NumberDisplay, Float, Int
from comfy_extras.nodes_mask import ImageColorToMask, SolidMask, ThresholdMask


def test_number_display_color_exists():
    """Verify that NumberDisplay has the color display mode option."""
    assert hasattr(NumberDisplay, "color")
    assert NumberDisplay.color.value == "color"


def test_int_and_float_serialization_with_color_display():
    """Verify that Int.Input and Float.Input display_mode serializes to string properly."""
    int_input = Int.Input("test_int", display_mode=NumberDisplay.color)
    float_input = Float.Input("test_float", display_mode=NumberDisplay.color)

    assert int_input.as_dict().get("display") == "color"
    assert float_input.as_dict().get("display") == "color"


def test_image_color_to_mask_schema_display_mode():
    """Verify that ImageColorToMask node schema specifies display_mode=color."""
    inputs = ImageColorToMask.INPUT_TYPES()
    assert "color" in inputs["required"]
    _, widget_dict = inputs["required"]["color"]
    assert widget_dict.get("display") == "color"


def test_solid_mask_schema_display_mode():
    """Verify that SolidMask node schema specifies display_mode=slider."""
    inputs = SolidMask.INPUT_TYPES()
    assert "value" in inputs["required"]
    _, widget_dict = inputs["required"]["value"]
    assert widget_dict.get("display") == "slider"


def test_threshold_mask_schema_display_mode():
    """Verify that ThresholdMask node schema specifies display_mode=slider."""
    inputs = ThresholdMask.INPUT_TYPES()
    assert "value" in inputs["required"]
    _, widget_dict = inputs["required"]["value"]
    assert widget_dict.get("display") == "slider"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
