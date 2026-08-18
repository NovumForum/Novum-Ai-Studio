import pytest
import torch
import time
from comfy_extras.nodes_mask import FeatherMask

def feather_old_reference(mask, left, top, right, bottom):
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


def test_feather_mask_mathematical_equivalence():
    torch.manual_seed(42)
    mask = torch.rand((2, 256, 256), dtype=torch.float32)

    test_cases = [
        (0, 0, 0, 0),
        (10, 0, 0, 0),
        (0, 15, 0, 0),
        (0, 0, 20, 0),
        (0, 0, 0, 25),
        (1, 1, 1, 1),
        (50, 40, 30, 20),
        (256, 256, 256, 256),
        (300, 300, 300, 300),  # oversized margins exceeding mask dims
    ]

    for left, top, right, bottom in test_cases:
        expected = feather_old_reference(mask, left, top, right, bottom)
        res = FeatherMask.execute(mask, left, top, right, bottom)
        actual = res.as_tuple()[0]
        assert torch.allclose(expected, actual, atol=1e-6), f"Mismatch for params ({left}, {top}, {right}, {bottom})"


def test_feather_mask_batch_and_shapes():
    mask_single = torch.ones((100, 100), dtype=torch.float32)
    res_single = FeatherMask.execute(mask_single, 10, 10, 10, 10).as_tuple()[0]
    assert res_single.shape == (1, 100, 100)

    mask_batch = torch.ones((4, 64, 128), dtype=torch.float32)
    res_batch = FeatherMask.execute(mask_batch, 5, 10, 15, 20).as_tuple()[0]
    assert res_batch.shape == (4, 64, 128)


def test_feather_mask_performance():
    mask = torch.ones((8, 512, 512), dtype=torch.float32)
    left, top, right, bottom = 100, 100, 100, 100

    t0 = time.perf_counter()
    res_old = feather_old_reference(mask, left, top, right, bottom)
    t1 = time.perf_counter()
    time_old = t1 - t0

    t2 = time.perf_counter()
    res_new = FeatherMask.execute(mask, left, top, right, bottom).as_tuple()[0]
    t3 = time.perf_counter()
    time_new = t3 - t2

    assert torch.allclose(res_old, res_new, atol=1e-6)
    # Ensure performance gain is achieved
    speedup = time_old / time_new
    assert speedup > 5.0, f"Expected speedup > 5x, got {speedup:.2f}x"
