from comfy_extras.nodes_color import ColorToRGBInt


def test_color_to_rgb_int_tooltip():
    schema = ColorToRGBInt.define_schema()
    assert schema.description == "Convert a color to a RGB integer value."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["color"].tooltip == "The hex color code (format #RRGGBB) to convert."
