import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_rgba():
    # Batch of 4 RGBA images (64x64)
    batch_size = 4
    rgba_image = torch.rand(batch_size, 64, 64, 4)

    output = SplitImageWithAlpha.execute(rgba_image)
    rgb, mask = output[0], output[1]

    assert rgb.shape == (batch_size, 64, 64, 3)
    assert mask.shape == (batch_size, 64, 64)

    # Verify RGB extraction
    assert torch.allclose(rgb, rgba_image[..., :3])
    # Verify alpha inversion: mask = 1.0 - alpha
    assert torch.allclose(mask, 1.0 - rgba_image[..., 3])


def test_split_image_with_alpha_rgb():
    # Batch of 4 RGB images (64x64) without alpha channel
    batch_size = 4
    rgb_image = torch.rand(batch_size, 64, 64, 3)

    output = SplitImageWithAlpha.execute(rgb_image)
    rgb, mask = output[0], output[1]

    assert rgb.shape == (batch_size, 64, 64, 3)
    assert mask.shape == (batch_size, 64, 64)

    assert torch.allclose(rgb, rgb_image)
    # When no alpha channel exists, alpha defaults to 1.0, so mask = 1.0 - 1.0 = 0.0
    assert torch.allclose(mask, torch.zeros((batch_size, 64, 64)))


def test_join_image_with_alpha_matching_dimensions():
    batch_size = 4
    rgb_image = torch.rand(batch_size, 64, 64, 3)
    mask = torch.rand(batch_size, 64, 64)

    output = JoinImageWithAlpha.execute(rgb_image, mask)
    rgba = output[0]

    assert rgba.shape == (batch_size, 64, 64, 4)
    assert torch.allclose(rgba[..., :3], rgb_image)
    # JoinImageWithAlpha converts mask back to alpha via (1.0 - mask)
    assert torch.allclose(rgba[..., 3], 1.0 - mask)


def test_join_image_with_alpha_mismatched_dimensions():
    batch_size = 2
    rgb_image = torch.rand(batch_size, 64, 64, 3)
    mask = torch.rand(batch_size, 32, 32)  # Mismatched mask size

    output = JoinImageWithAlpha.execute(rgb_image, mask)
    rgba = output[0]

    assert rgba.shape == (batch_size, 64, 64, 4)
    assert torch.allclose(rgba[..., :3], rgb_image)


def test_join_image_with_alpha_mismatched_batch_size():
    # Image has batch size 4, mask has batch size 2
    rgb_image = torch.rand(4, 64, 64, 3)
    mask = torch.rand(2, 64, 64)

    output = JoinImageWithAlpha.execute(rgb_image, mask)
    rgba = output[0]

    # Batch size should be truncated to min(len(image), len(alpha)) = 2
    assert rgba.shape == (2, 64, 64, 4)
    assert torch.allclose(rgba[..., :3], rgb_image[:2])
    assert torch.allclose(rgba[..., 3], 1.0 - mask[:2])
