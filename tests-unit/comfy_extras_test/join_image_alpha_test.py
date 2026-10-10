import sys
from unittest.mock import MagicMock
sys.modules["comfy_kitchen"] = MagicMock()

import torch
from comfy_extras.nodes_compositing import JoinImageWithAlpha, resize_mask


def reference_join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference implementation of JoinImageWithAlpha using per-frame looping."""
    batch_size = min(len(image), len(alpha))
    out_images = []

    alpha_resized = 1.0 - resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:, :, :3], alpha_resized[i].unsqueeze(2)), dim=2))

    return torch.stack(out_images)


def test_join_image_with_alpha_single_batch():
    image = torch.rand(1, 64, 64, 3)
    alpha = torch.rand(1, 64, 64)

    res = JoinImageWithAlpha.execute(image, alpha)
    out_tensor = res[0]
    ref_tensor = reference_join_image_with_alpha(image, alpha)

    assert out_tensor.shape == (1, 64, 64, 4)
    assert torch.equal(out_tensor, ref_tensor)


def test_join_image_with_alpha_multi_batch():
    image = torch.rand(8, 128, 128, 3)
    alpha = torch.rand(8, 128, 128)

    res = JoinImageWithAlpha.execute(image, alpha)
    out_tensor = res[0]
    ref_tensor = reference_join_image_with_alpha(image, alpha)

    assert out_tensor.shape == (8, 128, 128, 4)
    assert torch.equal(out_tensor, ref_tensor)


def test_join_image_with_alpha_rgba_input():
    # Input image with 4 channels should slice first 3 channels and attach new alpha
    image = torch.rand(4, 32, 32, 4)
    alpha = torch.rand(4, 32, 32)

    res = JoinImageWithAlpha.execute(image, alpha)
    out_tensor = res[0]
    ref_tensor = reference_join_image_with_alpha(image, alpha)

    assert out_tensor.shape == (4, 32, 32, 4)
    assert torch.equal(out_tensor, ref_tensor)


def test_join_image_with_alpha_unequal_batch_sizes():
    # Test image batch size larger than alpha batch size
    image_large = torch.rand(6, 48, 48, 3)
    alpha_small = torch.rand(3, 48, 48)

    res1 = JoinImageWithAlpha.execute(image_large, alpha_small)
    out1 = res1[0]
    ref1 = reference_join_image_with_alpha(image_large, alpha_small)

    assert out1.shape == (3, 48, 48, 4)
    assert torch.equal(out1, ref1)

    # Test alpha batch size larger than image batch size
    image_small = torch.rand(2, 48, 48, 3)
    alpha_large = torch.rand(5, 48, 48)

    res2 = JoinImageWithAlpha.execute(image_small, alpha_large)
    out2 = res2[0]
    ref2 = reference_join_image_with_alpha(image_small, alpha_large)

    assert out2.shape == (2, 48, 48, 4)
    assert torch.equal(out2, ref2)
