import torch
import pytest
import comfy.cli_args

# Ensure CPU mode is set before loading modules
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_mask import FeatherMask

def feather_mask_reference(mask, left, top, right, bottom):
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
    mask = torch.ones((1, 64, 64), dtype=torch.float32)
    res = FeatherMask.execute(mask, 0, 0, 0, 0)
    output = res.args[0]
    assert torch.equal(output, mask)


def test_feather_mask_values():
    torch.manual_seed(42)
    mask = torch.rand((2, 128, 128), dtype=torch.float32)
    left, top, right, bottom = 20, 15, 25, 30

    res = FeatherMask.execute(mask, left, top, right, bottom)
    output = res.args[0]

    expected = feather_mask_reference(mask, left, top, right, bottom)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-6)


def test_feather_mask_large_bounds():
    mask = torch.ones((1, 32, 32), dtype=torch.float32)
    left, top, right, bottom = 100, 100, 100, 100

    res = FeatherMask.execute(mask, left, top, right, bottom)
    output = res.args[0]

    expected = feather_mask_reference(mask, left, top, right, bottom)
    assert torch.allclose(output, expected, atol=1e-6)
