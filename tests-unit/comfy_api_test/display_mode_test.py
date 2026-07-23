import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_api.latest import IO
from comfy_extras.nodes_mask import ImageColorToMask

def test_number_display_enum_color():
    # Verify NumberDisplay has color attribute
    assert hasattr(IO.NumberDisplay, "color")
    assert IO.NumberDisplay.color == "color"

def test_image_color_to_mask_schema_serialization():
    # Verify the ImageColorToMask's schema displays color correctly
    schema = ImageColorToMask.GET_SCHEMA()
    color_input = None
    for inp in schema.inputs:
        if inp.id == "color":
            color_input = inp
            break

    assert color_input is not None
    assert color_input.display_mode == IO.NumberDisplay.color

    # Verify standard serialization dictionary output
    serialized = color_input.as_dict()
    assert serialized.get("display") == "color"
