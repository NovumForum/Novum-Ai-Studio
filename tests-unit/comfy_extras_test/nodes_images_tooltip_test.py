import pytest
from unittest.mock import patch, MagicMock

# Mock nodes and server modules before importing the images nodes
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_images import ImageScaleToMaxDimension, ResizeAndPadImage

def test_image_scale_to_max_dimension_schema():
    schema = ImageScaleToMaxDimension.define_schema()

    # Assertions on basic properties
    assert schema.node_id == "ImageScaleToMaxDimension"
    assert schema.display_name == "Scale Image to Max Dimension"
    assert "largest dimension" in schema.description
    assert "scale to max" in schema.search_aliases

    # Find inputs
    inputs_dict = {inp.id: inp for inp in schema.inputs}

    assert "upscale_method" in inputs_dict
    assert "largest_size" in inputs_dict

    assert "interpolation algorithm" in inputs_dict["upscale_method"].tooltip
    assert "target size" in inputs_dict["largest_size"].tooltip

def test_resize_and_pad_image_schema():
    schema = ResizeAndPadImage.define_schema()

    assert schema.node_id == "ResizeAndPadImage"
    assert schema.display_name == "Resize and Pad Image"
    assert "proportionally" in schema.description
    assert "letterbox" in schema.search_aliases

    inputs_dict = {inp.id: inp for inp in schema.inputs}

    assert "target_width" in inputs_dict
    assert "target_height" in inputs_dict
    assert "padding_color" in inputs_dict
    assert "interpolation" in inputs_dict

    assert "target width" in inputs_dict["target_width"].tooltip
    assert "target height" in inputs_dict["target_height"].tooltip
    assert "background color" in inputs_dict["padding_color"].tooltip
    assert "interpolation algorithm" in inputs_dict["interpolation"].tooltip
