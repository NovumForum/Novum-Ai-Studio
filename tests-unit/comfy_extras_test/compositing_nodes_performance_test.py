import torch
import pytest
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_rgba():
    batch, height, width = 4, 64, 64
    image_rgba = torch.rand(batch, height, width, 4)

    node = SplitImageWithAlpha()
    output = node.execute(image_rgba)
    out_images = output[0]
    out_masks = output[1]

    assert out_images.shape == (batch, height, width, 3)
    assert out_masks.shape == (batch, height, width)

    # Check exact mathematical equivalence with reference
    expected_images = image_rgba[..., :3]
    expected_masks = 1.0 - image_rgba[..., 3]

    torch.testing.assert_close(out_images, expected_images)
    torch.testing.assert_close(out_masks, expected_masks)


def test_split_image_with_alpha_rgb():
    batch, height, width = 2, 32, 32
    image_rgb = torch.rand(batch, height, width, 3)

    node = SplitImageWithAlpha()
    output = node.execute(image_rgb)
    out_images = output[0]
    out_masks = output[1]

    assert out_images.shape == (batch, height, width, 3)
    assert out_masks.shape == (batch, height, width)

    # RGB image should produce zeros mask (alpha = 1.0 => mask = 1.0 - 1.0 = 0.0)
    torch.testing.assert_close(out_images, image_rgb)
    torch.testing.assert_close(out_masks, torch.zeros_like(out_masks))


def test_join_image_with_alpha_same_size():
    batch, height, width = 4, 64, 64
    image = torch.rand(batch, height, width, 3)
    alpha = torch.rand(batch, height, width)

    node = JoinImageWithAlpha()
    output = node.execute(image, alpha)
    out_image = output[0]

    assert out_image.shape == (batch, height, width, 4)

    expected_rgb = image[..., :3]
    expected_alpha = 1.0 - alpha

    torch.testing.assert_close(out_image[..., :3], expected_rgb)
    torch.testing.assert_close(out_image[..., 3], expected_alpha)


def test_join_image_with_alpha_mismatched_batch():
    image = torch.rand(5, 32, 32, 3)
    alpha = torch.rand(3, 32, 32)

    node = JoinImageWithAlpha()
    output = node.execute(image, alpha)
    out_image = output[0]

    # Resulting batch size should be min(5, 3) = 3
    assert out_image.shape == (3, 32, 32, 4)
