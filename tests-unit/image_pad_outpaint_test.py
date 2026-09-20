import torch
from unittest.mock import patch, MagicMock

# Mock torch.cuda memory stats and device methods to allow safe top-level import of nodes.py in CPU-only test environment
with patch("torch.cuda.is_available", return_value=False), \
     patch("torch.cuda.current_device", return_value=0), \
     patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=8 * 1024**3)), \
     patch("torch.cuda.memory_stats", return_value={"reserved_bytes.all.current": 0}), \
     patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)):
    from nodes import ImagePadForOutpaint


def test_image_pad_for_outpaint_basic():
    node = ImagePadForOutpaint()
    image = torch.randn(1, 64, 64, 3)
    left, top, right, bottom, feathering = 16, 16, 16, 16, 10

    new_image, mask = node.expand_image(image, left, top, right, bottom, feathering)

    assert new_image.shape == (1, 64 + 16 + 16, 64 + 16 + 16, 3)
    assert mask.shape == (1, 64 + 16 + 16, 64 + 16 + 16)
    # Padded border area should be 1.0 in mask (outer padding)
    assert mask[0, 0, 0] == 1.0


def test_image_pad_for_outpaint_zero_feathering():
    node = ImagePadForOutpaint()
    image = torch.randn(1, 32, 32, 3)
    new_image, mask = node.expand_image(image, 8, 8, 8, 8, 0)

    assert new_image.shape == (1, 48, 48, 3)
    assert mask.shape == (1, 48, 48)
    # With feathering = 0, inner padded region mask values should be 0.0
    assert torch.all(mask[0, 8:40, 8:40] == 0.0)


def test_image_pad_for_outpaint_zero_padding():
    node = ImagePadForOutpaint()
    image = torch.randn(2, 64, 64, 3)
    # Test top=0, bottom=0, left=0, right=0 edge cases
    new_image, mask = node.expand_image(image, 0, 0, 0, 0, 10)

    assert new_image.shape == (2, 64, 64, 3)
    assert mask.shape == (1, 64, 64)


def test_image_pad_for_outpaint_feathering_edge_cases():
    node = ImagePadForOutpaint()
    image = torch.randn(1, 100, 100, 3)
    # Feathering too large (feathering * 2 >= d2 or d3) should skip feathering calculation safely
    new_image, mask = node.expand_image(image, 10, 10, 10, 10, 60)

    assert new_image.shape == (1, 120, 120, 3)
    assert mask.shape == (1, 120, 120)
