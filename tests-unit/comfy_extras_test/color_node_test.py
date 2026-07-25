from comfy_extras.nodes_color import ColorToRGBInt

def test_color_to_rgb_int_schema_tooltip():
    schema = ColorToRGBInt.define_schema()
    schema.finalize()
    schema.validate()

    # Find the "color" input in the inputs list
    color_input = None
    for inp in schema.inputs:
        if inp.id == "color":
            color_input = inp
            break

    assert color_input is not None
    assert color_input.tooltip == "Hex color string in #RRGGBB format (e.g., #FF0000 for red)."
