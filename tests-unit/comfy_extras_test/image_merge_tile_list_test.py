import torch
from comfy_extras.nodes_images import ImageMergeTileList, SplitImageToTileList

def test_image_merge_tile_list_execute():
    # Test multi-batch image split and merge with overlap
    b, h, w, c = 4, 1024, 1024, 3
    tile_w, tile_h, overlap = 512, 512, 64
    image = torch.randn((b, h, w, c))

    tiles = SplitImageToTileList.execute(image, tile_w, tile_h, overlap).output[0]
    out = ImageMergeTileList.execute(tiles, [w], [h], [overlap]).output[0]

    assert out.shape == image.shape
    # Check that reconstructed image is close to original image
    assert torch.allclose(out, image, atol=1e-4)

def test_image_merge_tile_list_no_overlap():
    # Test single-batch image split and merge with zero overlap
    b, h, w, c = 1, 512, 512, 3
    tile_w, tile_h, overlap = 256, 256, 0
    image = torch.randn((b, h, w, c))

    tiles = SplitImageToTileList.execute(image, tile_w, tile_h, overlap).output[0]
    out = ImageMergeTileList.execute(tiles, [w], [h], [overlap]).output[0]

    assert out.shape == image.shape
    assert torch.allclose(out, image, atol=1e-4)
