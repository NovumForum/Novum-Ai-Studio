from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_mask import SolidMask, InvertMask, ThresholdMask


class TestMaskNodeTooltip:
    def test_solid_mask_schema_properties(self):
        """Test that the SolidMask schema has the expected metadata and input tooltips."""
        schema = SolidMask.define_schema()

        assert schema.node_id == "SolidMask"
        assert schema.display_name == "Solid Mask"
        assert schema.description == "Generates a blank solid mask of a specified size filled with a constant intensity value."
        assert "blank mask" in schema.search_aliases
        assert "fill mask" in schema.search_aliases

        inputs_dict = {input_item.id: input_item for input_item in schema.inputs}
        assert "value" in inputs_dict
        assert "width" in inputs_dict
        assert "height" in inputs_dict

        assert inputs_dict["value"].tooltip == "The fill value/intensity of the mask. 0.0 is completely black (unmasked), 1.0 is completely white (masked)."
        assert inputs_dict["width"].tooltip == "The width of the generated mask in pixels."
        assert inputs_dict["height"].tooltip == "The height of the generated mask in pixels."

    def test_invert_mask_schema_properties(self):
        """Test that the InvertMask schema has the expected metadata and input tooltips."""
        schema = InvertMask.define_schema()

        assert schema.node_id == "InvertMask"
        assert schema.display_name == "Invert Mask"
        assert schema.description == "Inverts the mask values, swapping masked (white) areas with unmasked (black) areas."
        assert "reverse mask" in schema.search_aliases
        assert "negate mask" in schema.search_aliases

        inputs_dict = {input_item.id: input_item for input_item in schema.inputs}
        assert "mask" in inputs_dict
        assert inputs_dict["mask"].tooltip == "The input mask to invert."

    def test_threshold_mask_schema_properties(self):
        """Test that the ThresholdMask schema has the expected metadata and input tooltips."""
        schema = ThresholdMask.define_schema()

        assert schema.node_id == "ThresholdMask"
        assert schema.display_name == "Threshold Mask"
        assert schema.description == "Converts a soft or grayscale mask into a binary (black & white) mask based on a threshold value."
        assert "binary mask" in schema.search_aliases
        assert "binarize mask" in schema.search_aliases

        inputs_dict = {input_item.id: input_item for input_item in schema.inputs}
        assert "mask" in inputs_dict
        assert "value" in inputs_dict

        assert inputs_dict["mask"].tooltip == "The grayscale or soft input mask to binarize."
        assert inputs_dict["value"].tooltip == "The cutoff threshold. Values above this threshold become 1.0 (white/masked), while values below or equal become 0.0 (black/unmasked)."
