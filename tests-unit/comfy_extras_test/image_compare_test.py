from unittest.mock import patch, MagicMock
import torch

# Mock nodes and server module to avoid CUDA / server initialization during import
mock_nodes = MagicMock()
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_image_compare import ImageCompare, ImageCompareExtension


def test_image_compare_schema():
    schema = ImageCompare.define_schema()
    assert schema.node_id == "ImageCompare"
    assert schema.display_name == "Image Compare"
    assert schema.description == "Compares two images side by side with an interactive slider interface."
    assert "compare images" in schema.search_aliases
    assert "before after" in schema.search_aliases

    # Verify input tooltips
    input_a = next(i for i in schema.inputs if i.id == "image_a")
    assert input_a.tooltip == "First image or batch of images for visual comparison (left side)."

    input_b = next(i for i in schema.inputs if i.id == "image_b")
    assert input_b.tooltip == "Second image or batch of images for visual comparison (right side)."

    compare_view = next(i for i in schema.inputs if i.id == "compare_view")
    assert compare_view.tooltip == "Interactive slider widget configuration and view state."


def test_image_compare_execute():
    # Set up mock PreviewImage
    mock_preview = MagicMock()
    mock_preview.save_images.return_value = {"ui": {"images": [{"filename": "test.png", "subfolder": "", "type": "temp"}]}}
    mock_nodes.PreviewImage.return_value = mock_preview

    img_a = torch.zeros((1, 64, 64, 3))
    img_b = torch.ones((1, 64, 64, 3))

    res = ImageCompare.execute(image_a=img_a, image_b=img_b)
    assert "a_images" in res.ui
    assert "b_images" in res.ui
    assert len(res.ui["a_images"]) == 1
    assert len(res.ui["b_images"]) == 1

    # Test with None inputs
    res_empty = ImageCompare.execute(image_a=None, image_b=None)
    assert res_empty.ui["a_images"] == []
    assert res_empty.ui["b_images"] == []


import pytest


@pytest.mark.asyncio
async def test_image_compare_extension():
    ext = ImageCompareExtension()
    nodes_list = await ext.get_node_list()
    assert ImageCompare in nodes_list
