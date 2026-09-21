import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
from nodes import ImagePadForOutpaint

def test_expand_image_padding_dimensions():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 100, 100, 3)
    left, top, right, bottom = 10, 20, 30, 40

    padded_img, mask = node.expand_image(image, left, top, right, bottom, feathering=10)

    assert padded_img.shape == (1, 100 + top + bottom, 100 + left + right, 3)
    assert mask.shape == (1, 100 + top + bottom, 100 + left + right)
    # Check that original image is placed at correct offset
    assert torch.allclose(padded_img[:, top:top + 100, left:left + 100, :], image)

def test_expand_image_zero_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 64, 64, 3)
    left, top, right, bottom = 16, 16, 16, 16

    padded_img, mask = node.expand_image(image, left, top, right, bottom, feathering=0)

    # With feathering=0, inner mask portion should be all zeros
    inner_mask = mask[0, top:top + 64, left:left + 64]
    assert torch.allclose(inner_mask, torch.zeros(64, 64))

def test_expand_image_large_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 32, 32, 3)
    left, top, right, bottom = 16, 16, 16, 16

    # When feathering * 2 >= height or width, feathering condition is skipped
    padded_img, mask = node.expand_image(image, left, top, right, bottom, feathering=20)

    inner_mask = mask[0, top:top + 32, left:left + 32]
    assert torch.allclose(inner_mask, torch.zeros(32, 32))

def test_expand_image_feathering_values():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 100, 100, 3)
    left, top, right, bottom = 10, 10, 10, 10
    feathering = 20

    padded_img, mask = node.expand_image(image, left, top, right, bottom, feathering=feathering)

    inner_mask = mask[0, top:top + 100, left:left + 100]

    # At top-left corner (0,0), distance d is min(dt=0, db=100, dl=0, dr=100) = 0
    # v = (20 - 0) / 20 = 1.0 -> t = 1.0^2 = 1.0
    assert torch.isclose(inner_mask[0, 0], torch.tensor(1.0))

    # At center (50,50), distance d is 50 >= feathering (20), so t should be 0.0
    assert torch.isclose(inner_mask[50, 50], torch.tensor(0.0))

    # At distance d = 10 from top edge (row 10, col 50):
    # dt = 10, db = 90, dl = 50, dr = 50 -> min is 10
    # v = (20 - 10) / 20 = 0.5 -> t = 0.5^2 = 0.25
    assert torch.isclose(inner_mask[10, 50], torch.tensor(0.25))

def test_expand_image_asymmetric_padding():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 64, 64, 3)
    # top = 0, so dt is treated as height (64)
    padded_img, mask = node.expand_image(image, left=16, top=0, right=16, bottom=16, feathering=10)

    inner_mask = mask[0, 0:64, 16:16+64]
    # Row 0 has dt = 64 (since top=0), db = 64, dl = 30 (for col 30), dr = 34 -> min d = 30 >= 10 -> 0.0
    assert torch.isclose(inner_mask[0, 30], torch.tensor(0.0))
