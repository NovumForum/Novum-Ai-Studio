import pytest
import torch
from comfy_extras.nodes_mask import FeatherMask

def reference_feather(mask, left, top, right, bottom):
    output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()

    left = min(left, output.shape[-1])
    right = min(right, output.shape[-1])
    top = min(top, output.shape[-2])
    bottom = min(bottom, output.shape[-2])

    for x in range(left):
        feather_rate = (x + 1.0) / left
        output[:, :, x] *= feather_rate

    for x in range(right):
        feather_rate = (x + 1) / right
        output[:, :, -x] *= feather_rate

    for y in range(top):
        feather_rate = (y + 1) / top
        output[:, y, :] *= feather_rate

    for y in range(bottom):
        feather_rate = (y + 1) / bottom
        output[:, -y, :] *= feather_rate

    return output


def test_feather_mask_zero():
    mask = torch.ones((2, 64, 64), dtype=torch.float32)
    res = FeatherMask.execute(mask, 0, 0, 0, 0)
    output_tensor = res.args[0]
    assert torch.equal(output_tensor, mask)


def test_feather_mask_equivalence():
    test_cases = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (10, 0, 0, 0),
        (0, 15, 0, 0),
        (0, 0, 20, 0),
        (0, 0, 0, 25),
        (8, 12, 16, 20),
        (100, 100, 100, 100),  # bounds clamping case
    ]

    torch.manual_seed(42)
    mask = torch.rand((4, 64, 64), dtype=torch.float32)

    for left, top, right, bottom in test_cases:
        expected = reference_feather(mask, left, top, right, bottom)
        res = FeatherMask.execute(mask, left, top, right, bottom)
        actual = res.args[0]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_feather_mask_shape():
    mask = torch.ones((1, 32, 48), dtype=torch.float32)
    res = FeatherMask.execute(mask, 5, 5, 5, 5)
    actual = res.args[0]
    assert actual.shape == (1, 32, 48)
