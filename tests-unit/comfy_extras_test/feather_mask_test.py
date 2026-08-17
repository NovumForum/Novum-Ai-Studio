import torch
import pytest
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_mask import FeatherMask

def ref_feather(mask: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Reference implementation matching original unoptimized Python loop algorithm."""
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
    """Verify that vectorized FeatherMask produces exact mathematical output across different dimensions."""
    test_cases = [
        # (batch, height, width, left, top, right, bottom)
        (1, 64, 64, 10, 10, 10, 10),
        (2, 512, 512, 50, 20, 30, 40),
        (1, 128, 256, 0, 0, 0, 0),  # Zero feathering
        (1, 16, 16, 30, 30, 30, 30), # Margin larger than mask size
        (1, 1, 1, 5, 5, 5, 5),      # Single pixel mask
        (3, 100, 200, 1, 1, 1, 1),  # Margin of 1
    ]

    for batch, h, w, left, top, right, bottom in test_cases:
        torch.manual_seed(42)
        mask = torch.randn((batch, h, w), dtype=torch.float32)

        expected = ref_feather(mask, left, top, right, bottom)
        actual = FeatherMask.execute(mask, left, top, right, bottom).args[0]

        assert torch.allclose(expected, actual, atol=1e-6), f"Mismatch for case ({batch}, {h}, {w}, {left}, {top}, {right}, {bottom})"

def test_feather_mask_edge_cases():
    """Verify edge cases like zero dimensions and different shapes."""
    mask = torch.ones((1, 100, 100), dtype=torch.float32)

    # 1. Zero feathering returns untouched clone
    res = FeatherMask.execute(mask, 0, 0, 0, 0).args[0]
    assert torch.equal(res, mask)

    # 2. Feathering bounds exceeding dimensions
    res_large = FeatherMask.execute(mask, 200, 200, 200, 200).args[0]
    ref_large = ref_feather(mask, 200, 200, 200, 200)
    assert torch.allclose(res_large, ref_large, atol=1e-6)

def test_feather_mask_performance():
    """Verify that vectorized execution is significantly faster than python loop baseline."""
    import time

    mask = torch.ones((1, 1024, 1024), dtype=torch.float32)
    left, top, right, bottom = 256, 256, 256, 256

    # Time reference
    t0 = time.perf_counter()
    for _ in range(5):
        ref_feather(mask, left, top, right, bottom)
    t_ref = time.perf_counter() - t0

    # Time vectorized
    t1 = time.perf_counter()
    for _ in range(5):
        FeatherMask.execute(mask, left, top, right, bottom)
    t_vec = time.perf_counter() - t1

    speedup = t_ref / t_vec
    assert speedup > 5.0, f"Expected speedup > 5x, got {speedup:.2f}x"
