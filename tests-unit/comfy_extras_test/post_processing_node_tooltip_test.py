from comfy_extras.nodes_post_processing import (
    Blend,
    Blur,
    Quantize,
    Sharpen,
    ImageScaleToTotalPixels,
)


def test_post_processing_nodes_schema_metadata():
    nodes = [Blend, Blur, Quantize, Sharpen, ImageScaleToTotalPixels]

    for node in nodes:
        schema = node.define_schema()

        # Check display_name and description
        assert schema.display_name is not None and len(schema.display_name) > 0
        assert schema.description is not None and len(schema.description) > 0

        # Check search aliases
        assert isinstance(schema.search_aliases, list)
        assert len(schema.search_aliases) > 0

        # Check input tooltips
        for input_param in schema.inputs:
            assert getattr(input_param, "tooltip", None) is not None, f"Missing tooltip for input {input_param.name} in node {schema.node_id}"
            assert len(input_param.tooltip) > 0

        # Check output tooltips
        for output_param in schema.outputs:
            assert getattr(output_param, "tooltip", None) is not None, f"Missing tooltip for output in node {schema.node_id}"
            assert len(output_param.tooltip) > 0
