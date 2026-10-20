import torch
import pytest
from comfy_extras.nodes_compositing import SplitImageWithAlpha


def reference_split_image_with_alpha(image: torch.Tensor):
    """Reference unvectorized implementation for equivalence checking."""
    out_images = [i[:, :, :3] for i in image]
    out_alphas = [i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image]
    return torch.stack(out_images), 1.0 - torch.stack(out_alphas)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("height,width", [(64, 64), (128, 96)])
def test_split_image_with_alpha_rgba(batch_size, height, width):
    torch.manual_seed(42)
    image_rgba = torch.rand(batch_size, height, width, 4)

    output = SplitImageWithAlpha.execute(image_rgba)
    res_images, res_masks = output[0], output[1]

    ref_images, ref_masks = reference_split_image_with_alpha(image_rgba)

    assert res_images.shape == (batch_size, height, width, 3)
    assert res_masks.shape == (batch_size, height, width)

    torch.testing.assert_close(res_images, ref_images)
    torch.testing.assert_close(res_masks, ref_masks)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("height,width", [(64, 64), (128, 96)])
def test_split_image_with_alpha_rgb(batch_size, height, width):
    torch.manual_seed(42)
    image_rgb = torch.rand(batch_size, height, width, 3)

    output = SplitImageWithAlpha.execute(image_rgb)
    res_images, res_masks = output[0], output[1]

    ref_images, ref_masks = reference_split_image_with_alpha(image_rgb)

    assert res_images.shape == (batch_size, height, width, 3)
    assert res_masks.shape == (batch_size, height, width)

    # For RGB images, alpha mask should be all zeros (1.0 - 1.0 = 0.0)
    assert torch.all(res_masks == 0.0)

    torch.testing.assert_close(res_images, ref_images)
    torch.testing.assert_close(res_masks, ref_masks)


def test_split_image_with_alpha_device_and_dtype():
    torch.manual_seed(42)
    image_rgba = torch.rand(4, 32, 32, 4, dtype=torch.float32)

    output = SplitImageWithAlpha.execute(image_rgba)
    res_images, res_masks = output[0], output[1]

    assert res_images.device == image_rgba.device
    assert res_masks.device == image_rgba.device
    assert res_images.dtype == image_rgba.dtype
    assert res_masks.dtype == image_rgba.dtype
