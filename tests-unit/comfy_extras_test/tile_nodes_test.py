import torch
from unittest.mock import patch, MagicMock

# Mock nodes, server, and comfy.utils modules to prevent full model management initialization and CUDA/inductor checks
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_server = MagicMock()

mock_comfy_utils = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server, 'comfy.utils': mock_comfy_utils}):
    from comfy_extras.nodes_images import SplitImageToTileList, ImageMergeTileList

class TestTileNodes:
    def test_split_and_merge_tile_list_reconstruction(self):
        b, h, w, c = 1, 1024, 1024, 3
        tile_w, tile_h = 512, 512
        overlap = 64

        image = torch.full((b, h, w, c), 0.5, dtype=torch.float32)

        # Split image into tiles
        output = SplitImageToTileList.execute(image, tile_w, tile_h, overlap)
        tiles = output.args[0] if hasattr(output, "args") else output[0]
        assert isinstance(tiles, list)
        assert len(tiles) > 0

        # Merge tiles back into image
        merged_output = ImageMergeTileList.execute(tiles, [w], [h], [overlap])
        merged = merged_output.args[0] if hasattr(merged_output, "args") else merged_output[0]

        assert merged.shape == image.shape
        assert merged.dtype == image.dtype
        diff = torch.max(torch.abs(image - merged)).item()
        assert diff < 1e-4

    def test_merge_tile_list_no_overlap(self):
        b, h, w, c = 1, 512, 512, 3
        tile_w, tile_h = 256, 256
        overlap = 0

        image = torch.full((b, h, w, c), 0.5, dtype=torch.float32)

        output = SplitImageToTileList.execute(image, tile_w, tile_h, overlap)
        tiles = output.args[0] if hasattr(output, "args") else output[0]

        merged_output = ImageMergeTileList.execute(tiles, [w], [h], [overlap])
        merged = merged_output.args[0] if hasattr(merged_output, "args") else merged_output[0]

        assert merged.shape == image.shape
        diff = torch.max(torch.abs(image - merged)).item()
        assert diff < 1e-5
