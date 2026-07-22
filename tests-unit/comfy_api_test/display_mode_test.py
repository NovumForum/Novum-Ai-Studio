from comfy_api.latest import IO


def test_number_display_enum_has_color():
    # Verify that 'color' is a valid option in NumberDisplay enum
    assert hasattr(IO.NumberDisplay, "color")
    assert IO.NumberDisplay.color == "color"


def test_image_color_to_mask_schema_uses_color_display_mode(monkeypatch):
    # Set args.cpu = True before importing nodes or comfy extras to prevent CUDA-related errors
    from comfy.cli_args import args
    monkeypatch.setattr(args, "cpu", True)

    from comfy_extras.nodes_mask import ImageColorToMask
    schema = ImageColorToMask.define_schema()

    # Extract the schema inputs
    color_input = None
    for input_param in schema.inputs:
        if input_param.id == "color":
            color_input = input_param
            break

    assert color_input is not None
    # Check that display mode is serialized as "color" in dict representation
    serialized_input = color_input.as_dict()
    assert serialized_input.get("display") == "color"
