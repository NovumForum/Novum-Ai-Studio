from unittest.mock import patch, MagicMock

# Set CPU mode to avoid CUDA driver check errors on CPU-only test environments
from comfy.cli_args import args
args.cpu = True

# Mock nodes module to prevent CUDA and other dependency initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_post_processing import Blend


def test_image_blend_node_schema_tooltips():
    """Test that the ImageBlend schema defines the correct tooltips, search aliases, and node description."""
    # Obtain the finalized schema
    schema = Blend.GET_SCHEMA()

    # Check node id and basic properties
    assert schema.node_id == "ImageBlend"
    assert schema.display_name == "Image Blend"
    assert schema.category == "image/postprocessing"
    assert schema.description == "Blend two images together using a selectable blend mode and factor."

    # Check search aliases
    expected_aliases = ["blend", "image blend", "mix images", "overlay", "merge images", "combine images"]
    assert schema.search_aliases == expected_aliases

    # Find the input objects by id
    inputs_by_id = {inp.id: inp for inp in schema.inputs}

    assert "image1" in inputs_by_id
    assert inputs_by_id["image1"].tooltip == "The base background image (image blend target 1)."

    assert "image2" in inputs_by_id
    assert inputs_by_id["image2"].tooltip == "The foreground image to blend onto the base image (image blend target 2)."

    assert "blend_factor" in inputs_by_id
    assert inputs_by_id["blend_factor"].tooltip == (
        "Controls the blend intensity. At 0.0, only the base image (image1) is visible. "
        "At 1.0, the foreground image (image2) is fully blended using the selected blend mode."
    )

    assert "blend_mode" in inputs_by_id
    assert inputs_by_id["blend_mode"].tooltip == (
        "The mathematical formula used to combine the colors of both images "
        "(e.g., normal, multiply, screen, overlay, soft_light, difference)."
    )
