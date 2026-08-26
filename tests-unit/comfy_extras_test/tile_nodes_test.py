import torch
import pytest
from comfy_extras.nodes_images import SplitImageToTileList, ImageMergeTileList

def test_split_and_merge_tile_list():
    # Create test image tensor: [1, 512, 512, 3]
    image = torch.rand(1, 512, 512, 3)

    # Test splitting into 256x256 tiles with 64 overlap
    split_res = SplitImageToTileList.execute(image, tile_width=256, tile_height=256, overlap=64)
    tiles = split_res.args[0]
    assert len(tiles) > 0

    # Test merging tiles back to 512x512 image
    merge_res = ImageMergeTileList.execute(tiles, final_width=[512], final_height=[512], overlap=[64])
    merged = merge_res.args[0]

    assert merged.shape == image.shape
    # Check that reconstructed image is close to original image
    torch.testing.assert_close(merged, image, atol=1e-3, rtol=1e-3)

def test_merge_zero_overlap():
    image = torch.rand(1, 512, 512, 3)
    split_res = SplitImageToTileList.execute(image, tile_width=256, tile_height=256, overlap=0)
    tiles = split_res.args[0]

    merge_res = ImageMergeTileList.execute(tiles, final_width=[512], final_height=[512], overlap=[0])
    merged = merge_res.args[0]

    assert merged.shape == image.shape
    torch.testing.assert_close(merged, image, atol=1e-5, rtol=1e-5)
