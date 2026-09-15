import unittest.mock as mock
import torch

# Mock CUDA properties for CPU-only environments before importing nodes
mock.patch("torch.cuda.is_available", return_value=False).start()
mock.patch("torch.cuda.current_device", return_value=0).start()
mock.patch("torch.cuda.get_device_properties", return_value=mock.MagicMock(total_memory=8 * 1024**3)).start()
mock.patch("torch.cuda.memory_stats", return_value={"reserved_bytes.all.current": 0, "allocated_bytes.all.current": 0}).start()
mock.patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)).start()

from nodes import ImagePadForOutpaint


def test_image_pad_for_outpaint_basic():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 64, 64, 3))
    left, top, right, bottom, feathering = 16, 16, 16, 16, 10

    out_image, out_mask = node.expand_image(image, left, top, right, bottom, feathering)

    assert out_image.shape == (1, 64 + top + bottom, 64 + left + right, 3)
    assert out_mask.shape == (1, 64 + top + bottom, 64 + left + right)

    # Check that original image content is placed at (top, left)
    torch.testing.assert_close(
        out_image[:, top : top + 64, left : left + 64, :],
        image,
    )


def test_image_pad_for_outpaint_zero_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 32, 32, 3))
    left, top, right, bottom, feathering = 10, 10, 10, 10, 0

    out_image, out_mask = node.expand_image(image, left, top, right, bottom, feathering)

    assert out_image.shape == (1, 52, 52, 3)
    assert out_mask.shape == (1, 52, 52)
    # Mask inside original area should be all zeros when feathering is 0
    torch.testing.assert_close(out_mask[:, top : top + 32, left : left + 32], torch.zeros((1, 32, 32)))


def test_image_pad_for_outpaint_asymmetrical_and_zero_pads():
    node = ImagePadForOutpaint()
    image = torch.rand((2, 64, 64, 3))
    left, top, right, bottom, feathering = 0, 10, 20, 0, 12

    out_image, out_mask = node.expand_image(image, left, top, right, bottom, feathering)

    assert out_image.shape == (2, 74, 84, 3)
    assert out_mask.shape == (1, 74, 84)


def test_image_pad_for_outpaint_large_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand((1, 32, 32, 3))
    left, top, right, bottom, feathering = 10, 10, 10, 10, 500  # feathering * 2 >= d2

    out_image, out_mask = node.expand_image(image, left, top, right, bottom, feathering)

    assert out_image.shape == (1, 52, 52, 3)
    assert out_mask.shape == (1, 52, 52)
