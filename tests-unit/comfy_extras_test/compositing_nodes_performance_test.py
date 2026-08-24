import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_rgba():
    # Test SplitImageWithAlpha with 4-channel (RGBA) image batch
    batch_size, h, w = 4, 64, 64
    rgb_data = torch.rand(batch_size, h, w, 3)
    alpha_data = torch.rand(batch_size, h, w)
    rgba_image = torch.cat((rgb_data, alpha_data.unsqueeze(-1)), dim=-1)

    out_image, out_mask = SplitImageWithAlpha.execute(rgba_image).args

    # Verify shapes and dtypes
    assert out_image.shape == (batch_size, h, w, 3)
    assert out_mask.shape == (batch_size, h, w)
    assert out_image.dtype == rgba_image.dtype
    assert out_mask.dtype == rgba_image.dtype

    # Verify exact values
    assert torch.allclose(out_image, rgb_data)
    assert torch.allclose(out_mask, 1.0 - alpha_data)


def test_split_image_with_alpha_rgb():
    # Test SplitImageWithAlpha with 3-channel (RGB) image batch
    batch_size, h, w = 2, 32, 32
    rgb_image = torch.rand(batch_size, h, w, 3)

    out_image, out_mask = SplitImageWithAlpha.execute(rgb_image).args

    assert out_image.shape == (batch_size, h, w, 3)
    assert out_mask.shape == (batch_size, h, w)
    assert torch.allclose(out_image, rgb_image)
    assert torch.allclose(out_mask, torch.zeros((batch_size, h, w)))


def test_join_image_with_alpha():
    # Test JoinImageWithAlpha joining RGB images with alpha masks
    batch_size, h, w = 3, 64, 64
    rgb_image = torch.rand(batch_size, h, w, 3)
    alpha_mask = torch.rand(batch_size, h, w)

    out_rgba = JoinImageWithAlpha.execute(rgb_image, alpha_mask).args[0]

    assert out_rgba.shape == (batch_size, h, w, 4)
    assert out_rgba.dtype == rgb_image.dtype
    assert torch.allclose(out_rgba[:, :, :, :3], rgb_image)
    assert torch.allclose(out_rgba[:, :, :, 3], 1.0 - alpha_mask)


def test_join_image_with_alpha_mismatched_sizes():
    # Test JoinImageWithAlpha when alpha mask requires interpolation resize
    batch_size = 2
    rgb_image = torch.rand(batch_size, 64, 64, 3)
    alpha_mask = torch.rand(batch_size, 32, 32)  # different size

    out_rgba = JoinImageWithAlpha.execute(rgb_image, alpha_mask).args[0]

    assert out_rgba.shape == (batch_size, 64, 64, 4)
    assert torch.allclose(out_rgba[:, :, :, :3], rgb_image)
