from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_canny import Canny


class TestCannyTooltip:
    def test_canny_schema_properties(self):
        """Test that the Canny schema has the expected description, search aliases, and input tooltips."""
        schema = Canny.define_schema()

        # Check metadata
        assert schema.node_id == "Canny"
        assert schema.display_name == "Canny"
        assert "contour detection" in schema.search_aliases
        assert "edge detection" in schema.search_aliases
        assert schema.description == "Applies Canny edge detection to extract outlines, boundaries, and contours from the input image."

        # Verify inputs and their tooltips
        input_dict = {input_item.id: input_item for input_item in schema.inputs}

        assert "image" in input_dict
        assert input_dict["image"].tooltip == "The input image to extract edges from."

        assert "low_threshold" in input_dict
        assert input_dict["low_threshold"].tooltip == "The lower threshold for edge hysteresis. Lower values detect faint/soft edges."

        assert "high_threshold" in input_dict
        assert input_dict["high_threshold"].tooltip == "The upper threshold for edge hysteresis. Lower values include more edge segments."
