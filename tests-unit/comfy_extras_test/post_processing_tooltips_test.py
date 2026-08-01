from unittest.mock import patch, MagicMock

# Force CPU mode before model_management is imported to prevent CUDA errors
import comfy.cli_args
comfy.cli_args.args.cpu = True

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_post_processing import Blur, Quantize, Sharpen


def test_blur_schema_tooltips():
    schema = Blur.define_schema()

    # Assert node description and search aliases
    assert schema.description == "Blurs an image using a Gaussian filter."
    assert "gaussian blur" in schema.search_aliases
    assert "blur image" in schema.search_aliases

    # Check input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_dict
    assert inputs_dict["image"].tooltip == "The input image to blur."

    assert "blur_radius" in inputs_dict
    assert "Higher values increase" in inputs_dict["blur_radius"].tooltip

    assert "sigma" in inputs_dict
    assert "The sigma value" in inputs_dict["sigma"].tooltip


def test_quantize_schema_tooltips():
    schema = Quantize.define_schema()

    # Assert node description and search aliases
    assert "Reduces the number of unique colors" in schema.description
    assert "dither" in schema.search_aliases
    assert "retro" in schema.search_aliases

    # Check input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_dict
    assert inputs_dict["image"].tooltip == "The input image to quantize."

    assert "colors" in inputs_dict
    assert "maximum number of colors" in inputs_dict["colors"].tooltip

    assert "dither" in inputs_dict
    assert "The dithering algorithm" in inputs_dict["dither"].tooltip


def test_sharpen_schema_tooltips():
    schema = Sharpen.define_schema()

    # Assert node description and search aliases
    assert "Sharpens an image by enhancing" in schema.description
    assert "unsharp mask" in schema.search_aliases
    assert "crisp" in schema.search_aliases

    # Check input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_dict
    assert inputs_dict["image"].tooltip == "The input image to sharpen."

    assert "sharpen_radius" in inputs_dict
    assert "radius of the sharpening" in inputs_dict["sharpen_radius"].tooltip

    assert "sigma" in inputs_dict
    assert "Gaussian kernel used to generate" in inputs_dict["sigma"].tooltip

    assert "alpha" in inputs_dict
    assert "strength of the sharpening" in inputs_dict["alpha"].tooltip
