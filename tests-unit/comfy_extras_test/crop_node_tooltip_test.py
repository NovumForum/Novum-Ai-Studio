import pytest
from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_images import ImageCropV2, BoundingBox


class TestCropAndBoundingBoxUX:

    def test_image_crop_v2_schema_ux(self):
        """Test ImageCropV2 schema description, search aliases, and input tooltips."""
        schema = ImageCropV2.GET_SCHEMA()

        # Verify node-level description and search aliases
        assert ImageCropV2.DESCRIPTION == "Crops an image using a specified bounding box region. This helps focus on specific objects or regions of interest."
        assert "trim" in schema.search_aliases
        assert "slice" in schema.search_aliases
        assert "cut" in schema.search_aliases
        assert "crop image" in schema.search_aliases
        assert "region of interest" in schema.search_aliases

        # Verify input parameter-level tooltips
        input_types = ImageCropV2.INPUT_TYPES()
        assert "required" in input_types

        # image input tooltip
        assert "image" in input_types["required"]
        assert input_types["required"]["image"][1]["tooltip"] == "The source image or batch of images to crop."

        # crop_region input tooltip
        assert "crop_region" in input_types["required"]
        assert input_types["required"]["crop_region"][1]["tooltip"] == "The bounding box defining the x, y coordinates and width, height of the area to crop."

    def test_bounding_box_schema_ux(self):
        """Test BoundingBox schema description, search aliases, and input tooltips."""
        schema = BoundingBox.GET_SCHEMA()

        # Verify node-level description and search aliases
        assert BoundingBox.DESCRIPTION == "Defines a rectangular bounding box with coordinates (x, y) and dimensions (width, height) to specify regions of interest for image operations."
        assert "crop region" in schema.search_aliases
        assert "bbox" in schema.search_aliases
        assert "rectangle" in schema.search_aliases
        assert "area selection" in schema.search_aliases
        assert "dimensions" in schema.search_aliases

        # Verify input parameter-level tooltips
        input_types = BoundingBox.INPUT_TYPES()
        assert "required" in input_types

        # x input tooltip
        assert "x" in input_types["required"]
        assert input_types["required"]["x"][1]["tooltip"] == "The horizontal (X) starting coordinate of the bounding box."

        # y input tooltip
        assert "y" in input_types["required"]
        assert input_types["required"]["y"][1]["tooltip"] == "The vertical (Y) starting coordinate of the bounding box."

        # width input tooltip
        assert "width" in input_types["required"]
        assert input_types["required"]["width"][1]["tooltip"] == "The horizontal width of the bounding box in pixels."

        # height input tooltip
        assert "height" in input_types["required"]
        assert input_types["required"]["height"][1]["tooltip"] == "The vertical height of the bounding box in pixels."
