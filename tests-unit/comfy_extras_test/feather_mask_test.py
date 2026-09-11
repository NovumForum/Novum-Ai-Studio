import torch
from unittest.mock import MagicMock

# Mock CUDA functions before comfy import on CPU-only test runners
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda dev: MagicMock(total_memory=8 * 1024 * 1024 * 1024)
torch.cuda.memory_stats = lambda dev=None: {
    'allocated_bytes.all.current': 0,
    'reserved_bytes.all.current': 0,
    'inactive_split_bytes.all.current': 0,
}
torch.cuda.mem_get_info = lambda dev=None: (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)

import pytest
from comfy_extras.nodes_mask import FeatherMask

def legacy_feather_mask(mask, left, top, right, bottom):
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
    mask = torch.ones((1, 32, 32), dtype=torch.float32)
    res = FeatherMask.execute(mask, 0, 0, 0, 0).args[0]
    assert torch.equal(res, mask)


def test_feather_mask_clamped_bounds():
    mask = torch.ones((1, 10, 10), dtype=torch.float32)
    res = FeatherMask.execute(mask, 20, 20, 20, 20).args[0]
    res_leg = legacy_feather_mask(mask, 20, 20, 20, 20)
    assert torch.allclose(res, res_leg, atol=1e-5)


def test_feather_mask_batch_processing():
    mask = torch.rand((3, 64, 64), dtype=torch.float32)
    res = FeatherMask.execute(mask, 5, 10, 15, 20).args[0]
    res_leg = legacy_feather_mask(mask, 5, 10, 15, 20)
    assert torch.allclose(res, res_leg, atol=1e-5)


@pytest.mark.parametrize(
    "shape, params",
    [
        ((1, 16, 16), (1, 1, 1, 1)),
        ((1, 30, 40), (5, 10, 15, 20)),
        ((2, 100, 100), (0, 20, 0, 30)),
        ((1, 50, 50), (25, 0, 25, 0)),
        ((2, 128, 128), (64, 64, 64, 64)),
    ],
)
def test_feather_mask_equivalence(shape, params):
    mask = torch.rand(shape, dtype=torch.float32)
    left, top, right, bottom = params
    res_opt = FeatherMask.execute(mask, left, top, right, bottom).args[0]
    res_leg = legacy_feather_mask(mask, left, top, right, bottom)
    assert torch.allclose(res_opt, res_leg, atol=1e-5)
