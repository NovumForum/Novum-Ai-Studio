import pytest
from comfy_extras.nodes_color import ColorToRGBInt


def test_color_to_rgb_int_schema():
    schema = ColorToRGBInt.define_schema()
    assert schema.node_id == "ColorToRGBInt"
    assert schema.display_name == "Color to RGB Int"
    assert schema.category == "utils"
    assert schema.description == "Convert a color to a RGB integer value."
    assert schema.search_aliases == ["color to int", "hex to rgb", "color code", "color conversion", "rgb int", "hex color"]

    # Inputs
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "color" in inputs_by_id
    assert inputs_by_id["color"].tooltip == "Hex color string in #RRGGBB format to convert to integer."

    # Outputs
    outputs = schema.outputs
    assert len(outputs) == 1
    assert outputs[0].display_name == "rgb_int"
    assert outputs[0].tooltip == "Integer representation of the RGB color (0xRRGGBB)."


def test_color_to_rgb_int_execute_valid():
    res = ColorToRGBInt.execute("#FF0000")
    assert res.args[0] == 16711680

    res = ColorToRGBInt.execute("#00FF00")
    assert res.args[0] == 65280

    res = ColorToRGBInt.execute("#0000FF")
    assert res.args[0] == 255

    res = ColorToRGBInt.execute("#FFFFFF")
    assert res.args[0] == 16777215


def test_color_to_rgb_int_execute_invalid():
    with pytest.raises(ValueError, match="Color must be in format #RRGGBB"):
        ColorToRGBInt.execute("FF0000")

    with pytest.raises(ValueError, match="Color must be in format #RRGGBB"):
        ColorToRGBInt.execute("#FF000")
