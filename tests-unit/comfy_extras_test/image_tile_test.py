import torch
import pytest
from unittest.mock import patch, MagicMock

# Mock nodes and server modules to prevent CUDA and server initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_images import SplitImageToTileList, ImageMergeTileList


def test_tile_split_and_merge_zero_overlap():
    gen = torch.Generator().manual_seed(42)
    # 2 batch items, 512x512 image, 256x256 tiles, 0 overlap
    image = torch.rand((2, 512, 512, 3), generator=gen)

    split_node = SplitImageToTileList()
    split_out = split_node.execute(image, tile_width=256, tile_height=256, overlap=0)
    tile_list = split_out.args[0]

    assert len(tile_list) == 4
    for tile in tile_list:
        assert tile.shape == (2, 256, 256, 3)

    merge_node = ImageMergeTileList()
    merge_out = merge_node.execute(tile_list, final_width=[512], final_height=[512], overlap=[0])
    reconstructed = merge_out.args[0]

    assert reconstructed.shape == (2, 512, 512, 3)
    assert torch.allclose(reconstructed, image, atol=1e-6)


def test_tile_split_and_merge_with_overlap():
    gen = torch.Generator().manual_seed(42)
    # Single batch item, 512x512 image, 256x256 tiles, 64 overlap
    image = torch.rand((1, 512, 512, 3), generator=gen)

    split_node = SplitImageToTileList()
    split_out = split_node.execute(image, tile_width=256, tile_height=256, overlap=64)
    tile_list = split_out.args[0]

    assert len(tile_list) > 1

    merge_node = ImageMergeTileList()
    merge_out = merge_node.execute(tile_list, final_width=[512], final_height=[512], overlap=[64])
    reconstructed = merge_out.args[0]

    assert reconstructed.shape == (1, 512, 512, 3)
    assert not torch.isnan(reconstructed).any()


def test_tile_split_and_merge_multi_batch():
    gen = torch.Generator().manual_seed(123)
    # 4 batch items, non-power-of-two dimensions (600x450), 200x200 tiles, 32 overlap
    image = torch.rand((4, 450, 600, 3), generator=gen)

    split_node = SplitImageToTileList()
    split_out = split_node.execute(image, tile_width=200, tile_height=200, overlap=32)
    tile_list = split_out.args[0]

    merge_node = ImageMergeTileList()
    merge_out = merge_node.execute(tile_list, final_width=[600], final_height=[450], overlap=[32])
    reconstructed = merge_out.args[0]

    assert reconstructed.shape == (4, 450, 600, 3)
    assert not torch.isnan(reconstructed).any()
