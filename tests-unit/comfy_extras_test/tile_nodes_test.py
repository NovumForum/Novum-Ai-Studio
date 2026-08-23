import sys
import unittest.mock
# Mock server module if required during import chain
sys.modules['server'] = unittest.mock.MagicMock()

import torch
import comfy.cli_args
comfy.cli_args.args = comfy.cli_args.parser.parse_args(['--cpu'])

from comfy_extras.nodes_images import SplitImageToTileList, ImageMergeTileList

def test_split_and_merge_tile_list_reconstruction():
    image = torch.rand((1, 512, 512, 3), dtype=torch.float32)
    split_res = SplitImageToTileList.execute(image, tile_width=256, tile_height=256, overlap=64)
    tile_list = split_res[0]

    assert len(tile_list) > 1

    merge_res = ImageMergeTileList.execute(tile_list, final_width=[512], final_height=[512], overlap=[64])
    merged = merge_res[0]

    assert merged.shape == image.shape
    assert torch.allclose(image, merged, atol=1e-4)

def test_tile_list_no_overlap():
    image = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    split_res = SplitImageToTileList.execute(image, tile_width=128, tile_height=128, overlap=0)
    tile_list = split_res[0]

    assert len(tile_list) == 4

    merge_res = ImageMergeTileList.execute(tile_list, final_width=[256], final_height=[256], overlap=[0])
    merged = merge_res[0]

    assert merged.shape == image.shape
    assert torch.allclose(image, merged, atol=1e-4)
