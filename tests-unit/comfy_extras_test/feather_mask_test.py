import pytest
import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_mask import FeatherMask

def test_feather_mask_zero_expansion():
    mask = torch.ones((2, 64, 64), dtype=torch.float32)
    res = FeatherMask.execute(mask, 0, 0, 0, 0)
    assert res is not None
    out = res[0]
    assert torch.equal(out, mask)

def test_feather_mask_boundary_values():
    mask = torch.ones((1, 100, 100), dtype=torch.float32)
    res = FeatherMask.execute(mask, 10, 15, 20, 25)
    out = res[0]
    assert out.shape == (1, 100, 100)

    # Reference calculation matching exact legacy algorithm loop
    legacy_out = mask.clone()
    left, top, right, bottom = 10, 15, 20, 25
    for x in range(left):
        legacy_out[:, :, x] *= (x + 1.0) / left
    for x in range(right):
        legacy_out[:, :, -x] *= (x + 1.0) / right
    for y in range(top):
        legacy_out[:, y, :] *= (y + 1.0) / top
    for y in range(bottom):
        legacy_out[:, -y, :] *= (y + 1.0) / bottom

    assert torch.allclose(out, legacy_out, atol=1e-6)

def test_feather_mask_clamped_bounds():
    mask = torch.ones((1, 20, 20), dtype=torch.float32)
    res = FeatherMask.execute(mask, left=50, top=50, right=50, bottom=50)
    out = res[0]
    assert out.shape == (1, 20, 20)
