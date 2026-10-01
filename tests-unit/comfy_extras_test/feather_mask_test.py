import pytest
import torch
import comfy.cli_args

# Ensure CPU mode is set when CUDA GPU driver is unavailable
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_mask import FeatherMask

def unvectorized_feather(mask, left, top, right, bottom):
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


def test_feather_mask_zero_feathering():
    mask = torch.ones((2, 64, 64), dtype=torch.float32)
    node_output = FeatherMask.execute(mask, 0, 0, 0, 0)
    res = node_output.args[0]
    assert torch.equal(res, mask)


def test_feather_mask_equivalence_with_reference():
    torch.manual_seed(42)
    mask = torch.rand((3, 128, 128), dtype=torch.float32)
    left, top, right, bottom = 15, 20, 10, 25

    node_output = FeatherMask.execute(mask, left, top, right, bottom)
    res_vec = node_output.args[0]
    res_ref = unvectorized_feather(mask, left, top, right, bottom)

    assert torch.allclose(res_vec, res_ref, atol=1e-6)


def test_feather_mask_oversized_bounds():
    mask = torch.ones((1, 30, 30), dtype=torch.float32)
    node_output = FeatherMask.execute(mask, 100, 100, 100, 100)
    res_vec = node_output.args[0]
    res_ref = unvectorized_feather(mask, 100, 100, 100, 100)

    assert res_vec.shape == (1, 30, 30)
    assert torch.allclose(res_vec, res_ref, atol=1e-6)


def test_feather_mask_individual_sides():
    mask = torch.ones((1, 10, 10), dtype=torch.float32)

    # Test left only
    res_left = FeatherMask.execute(mask, 5, 0, 0, 0).args[0]
    ref_left = unvectorized_feather(mask, 5, 0, 0, 0)
    assert torch.allclose(res_left, ref_left, atol=1e-6)

    # Test right only
    res_right = FeatherMask.execute(mask, 0, 0, 5, 0).args[0]
    ref_right = unvectorized_feather(mask, 0, 0, 5, 0)
    assert torch.allclose(res_right, ref_right, atol=1e-6)

    # Test top only
    res_top = FeatherMask.execute(mask, 0, 5, 0, 0).args[0]
    ref_top = unvectorized_feather(mask, 0, 5, 0, 0)
    assert torch.allclose(res_top, ref_top, atol=1e-6)

    # Test bottom only
    res_bottom = FeatherMask.execute(mask, 0, 0, 0, 5).args[0]
    ref_bottom = unvectorized_feather(mask, 0, 0, 0, 5)
    assert torch.allclose(res_bottom, ref_bottom, atol=1e-6)
