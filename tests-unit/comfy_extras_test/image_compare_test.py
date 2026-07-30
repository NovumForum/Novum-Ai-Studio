import torch
from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization and mock PreviewImage
mock_nodes = MagicMock()
mock_preview_instance = MagicMock()
mock_preview_instance.save_images.side_effect = lambda image, prefix: {"ui": {"images": [f"{prefix}_saved"]}}
mock_nodes.PreviewImage.return_value = mock_preview_instance

mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_image_compare import ImageCompare


class TestImageCompare:

    def test_schema(self):
        """Test that the ImageCompare schema is configured correctly with UX/accessibility features."""
        schema = ImageCompare.GET_SCHEMA()
        assert schema.node_id == "ImageCompare"
        assert schema.display_name == "Image Compare"
        assert "interactive comparison slider" in schema.description
        assert "compare" in schema.search_aliases
        assert "slider" in schema.search_aliases
        assert "side-by-side" in schema.search_aliases

        # Verify tooltips on inputs
        input_map = {i.id: i for i in schema.inputs}
        assert "image_a" in input_map
        assert "image_b" in input_map
        assert "compare_view" in input_map

        assert input_map["image_a"].tooltip == "The first image (left side) to compare."
        assert input_map["image_b"].tooltip == "The second image (right side) to compare."
        assert input_map["compare_view"].tooltip == "Interactive comparison view with slider control."

    def test_execute(self):
        """Test that execute saves images using the PreviewImage node and returns correct format."""
        image_a = torch.rand(1, 64, 64, 3)
        image_b = torch.rand(1, 64, 64, 3)

        mock_preview_instance.save_images.reset_mock()
        result = ImageCompare.execute(image_a=image_a, image_b=image_b)
        assert result.ui == {
            "a_images": ["comfy.compare.a_saved"],
            "b_images": ["comfy.compare.b_saved"]
        }

        # Check interaction with mock using torch.equal
        call_args = mock_preview_instance.save_images.call_args_list
        assert len(call_args) == 2

        # First call (image_a)
        assert torch.equal(call_args[0][0][0], image_a)
        assert call_args[0][0][1] == "comfy.compare.a"

        # Second call (image_b)
        assert torch.equal(call_args[1][0][0], image_b)
        assert call_args[1][0][1] == "comfy.compare.b"

    def test_execute_empty_optional(self):
        """Test that execute handles None or empty optional inputs gracefully."""
        result = ImageCompare.execute(image_a=None, image_b=None)
        assert result.ui == {
            "a_images": [],
            "b_images": []
        }
