import torch
import comfy.cli_args

# Set cpu mode for test environment
comfy.cli_args.args.cpu = True

from nodes import ImagePadForOutpaint


def test_image_pad_for_outpaint_basic():
    padder = ImagePadForOutpaint()
    image = torch.rand(1, 128, 128, 3)
    left, top, right, bottom, feathering = 32, 16, 24, 40, 20

    new_image, mask = padder.expand_image(image, left, top, right, bottom, feathering)

    assert new_image.shape == (1, 128 + top + bottom, 128 + left + right, 3)
    assert mask.shape == (1, 128 + top + bottom, 128 + left + right)
    # The original image content should be copied into the padded area
    torch.testing.assert_close(new_image[:, top:top + 128, left:left + 128, :], image)


def test_image_pad_for_outpaint_zero_feathering():
    padder = ImagePadForOutpaint()
    image = torch.ones(1, 64, 64, 3)
    left, top, right, bottom, feathering = 16, 16, 16, 16, 0

    new_image, mask = padder.expand_image(image, left, top, right, bottom, feathering)

    assert new_image.shape == (1, 96, 96, 3)
    assert mask.shape == (1, 96, 96)
    # Mask inside image bounds should be 0 (feathering = 0)
    torch.testing.assert_close(mask[0, top:top + 64, left:left + 64], torch.zeros(64, 64))


def test_image_pad_for_outpaint_feathering_exceeding_bounds():
    padder = ImagePadForOutpaint()
    image = torch.rand(1, 32, 32, 3)
    # feathering * 2 >= d2 (50 * 2 >= 32)
    left, top, right, bottom, feathering = 16, 16, 16, 16, 50

    new_image, mask = padder.expand_image(image, left, top, right, bottom, feathering)

    assert new_image.shape == (1, 64, 64, 3)
    assert mask.shape == (1, 64, 64)
    torch.testing.assert_close(mask[0, top:top + 32, left:left + 32], torch.zeros(32, 32))


def test_image_pad_for_outpaint_asymmetric_and_zero_sides():
    padder = ImagePadForOutpaint()
    image = torch.rand(2, 64, 64, 4)

    # Test top = 0, bottom = 16, left = 8, right = 0
    new_image, mask = padder.expand_image(image, left=8, top=0, right=0, bottom=16, feathering=10)

    assert new_image.shape == (2, 80, 72, 4)
    assert mask.shape == (1, 80, 72)


def test_image_pad_for_outpaint_numerical_correctness():
    padder = ImagePadForOutpaint()
    image = torch.rand(1, 64, 64, 3)
    left, top, right, bottom, feathering = 16, 16, 16, 16, 10

    new_image, mask = padder.expand_image(image, left, top, right, bottom, feathering)

    inner_mask = mask[0, top:top + 64, left:left + 64]

    # Center pixels (further than feathering = 10 from any edge) should be 0.0
    assert torch.allclose(inner_mask[10:54, 10:54], torch.tensor(0.0))

    # Corner pixel at (0, 0) of inner image has distance d = 0, so v = (10-0)/10 = 1.0, v^2 = 1.0
    assert torch.isclose(inner_mask[0, 0], torch.tensor(1.0))
