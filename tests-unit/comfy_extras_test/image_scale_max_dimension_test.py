from unittest.mock import MagicMock, patch
import torch

# Mock nodes and server modules to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_images import ImageScaleToMaxDimension


def test_image_scale_to_max_dimension_schema():
    schema = ImageScaleToMaxDimension.define_schema()
    assert schema.node_id == "ImageScaleToMaxDimension"
    assert schema.display_name == "Image Scale to Max Dimension"
    assert (
        schema.description
        == "Scales an image proportionally so its maximum dimension matches largest_size."
    )
    assert schema.search_aliases == [
        "upscale max",
        "resize largest dimension",
        "fit max size",
    ]

    # Verify input tooltips
    inputs = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs
    assert inputs["image"].tooltip == "The image to scale."
    assert "upscale_method" in inputs
    assert (
        inputs["upscale_method"].tooltip
        == "The resampling algorithm used for scaling the image."
    )
    assert "largest_size" in inputs
    assert (
        inputs["largest_size"].tooltip
        == "The target size for the larger dimension of the image."
    )

    # Verify output tooltips and display_name
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "IMAGE"
    assert schema.outputs[0].tooltip == "The scaled image."


def test_image_scale_to_max_dimension_execution():
    # Test scaling landscape image (width > height)
    image = torch.zeros((1, 100, 200, 3))  # H=100, W=200
    res = ImageScaleToMaxDimension.execute(image, "bilinear", 400)
    scaled_image = res.args[0]
    assert scaled_image.shape == (1, 200, 400, 3)

    # Test scaling portrait image (height > width)
    image_portrait = torch.zeros((1, 300, 150, 3))  # H=300, W=150
    res_portrait = ImageScaleToMaxDimension.execute(
        image_portrait, "nearest-exact", 600
    )
    scaled_portrait = res_portrait.args[0]
    assert scaled_portrait.shape == (1, 600, 300, 3)

    # Test scaling square image
    image_square = torch.zeros((1, 250, 250, 3))
    res_square = ImageScaleToMaxDimension.execute(image_square, "bicubic", 500)
    scaled_square = res_square.args[0]
    assert scaled_square.shape == (1, 500, 500, 3)
