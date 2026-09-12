import torch
from unittest.mock import MagicMock

# Mock torch.cuda functions so comfy.model_management imports on CPU-only machines without NVIDIA drivers
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda dev=None: MagicMock(total_memory=16 * 1024 * 1024 * 1024)
torch.cuda.memory_stats = lambda dev=None: {'reserved_bytes.all.current': 0}
torch.cuda.mem_get_info = lambda dev=None: (16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)

from nodes import ImagePadForOutpaint


def test_image_pad_for_outpaint_basic():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 100, 100, 3), dtype=torch.float32)

    new_image, mask = node.expand_image(image, left=10, top=20, right=30, bottom=40, feathering=10)

    # Output image shape: (1, 100 + 20 + 40, 100 + 10 + 30, 3) = (1, 160, 140, 3)
    assert new_image.shape == (1, 160, 140, 3)
    # Output mask shape: (1, 160, 140)
    assert mask.shape == (1, 160, 140)

    # Check padded region background color is 0.5
    assert torch.allclose(new_image[0, :20, :, :], torch.tensor(0.5))
    assert torch.allclose(new_image[0, 120:, :, :], torch.tensor(0.5))

    # Check center image matches input image
    assert torch.allclose(new_image[0, 20:120, 10:110, :], image[0])


def test_image_pad_for_outpaint_zero_padding():
    node = ImagePadForOutpaint()
    image = torch.rand((2, 64, 64, 3), dtype=torch.float32)

    new_image, mask = node.expand_image(image, left=0, top=0, right=0, bottom=0, feathering=10)

    assert new_image.shape == (2, 64, 64, 3)
    assert mask.shape == (1, 64, 64)
    assert torch.allclose(new_image, image)


def test_image_pad_for_outpaint_zero_feathering():
    node = ImagePadForOutpaint()
    image = torch.ones((1, 50, 50, 3), dtype=torch.float32)

    new_image, mask = node.expand_image(image, left=10, top=10, right=10, bottom=10, feathering=0)

    assert new_image.shape == (1, 70, 70, 3)
    assert mask.shape == (1, 70, 70)
    # With feathering=0, mask inside the image rectangle should be all zeros
    assert torch.allclose(mask[0, 10:60, 10:60], torch.tensor(0.0))
    # Outside should be ones
    assert torch.allclose(mask[0, :10, :], torch.tensor(1.0))


def test_image_pad_for_outpaint_large_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 30, 30, 3), dtype=torch.float32)

    # feathering * 2 >= d2 or d3 (100 * 2 >= 30)
    new_image, mask = node.expand_image(image, left=10, top=10, right=10, bottom=10, feathering=100)

    assert new_image.shape == (1, 50, 50, 3)
    assert mask.shape == (1, 50, 50)
    # When feathering is too large, feathering condition feathering*2 < d2 is False, so t remains 0
    assert torch.allclose(mask[0, 10:40, 10:40], torch.tensor(0.0))


def test_image_pad_for_outpaint_asymmetric_padding():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 80, 80, 3), dtype=torch.float32)

    # Top and left padded, bottom and right zero
    new_image, mask = node.expand_image(image, left=16, top=16, right=0, bottom=0, feathering=10)

    assert new_image.shape == (1, 96, 96, 3)
    assert mask.shape == (1, 96, 96)
    # Ensure no NaN or Inf in mask or image
    assert not torch.isnan(new_image).any()
    assert not torch.isnan(mask).any()
