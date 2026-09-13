import torch
from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization during import on CPU environments
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

with patch.dict('sys.modules', {'nodes': mock_nodes}):
    from comfy_extras.nodes_mask import FeatherMask

def original_feather_reference(mask, left, top, right, bottom):
    """Reference legacy implementation for numerical equivalence verification."""
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
    out = res[0]
    assert torch.allclose(mask, out)

def test_feather_mask_equivalence():
    shapes = [(1, 32, 32), (2, 64, 128), (4, 100, 100)]
    param_sets = [
        (10, 10, 10, 10),
        (5, 0, 15, 0),
        (0, 20, 0, 25),
        (1, 1, 1, 1),
        (50, 50, 50, 50),
    ]

    for shape in shapes:
        for l, t, r, b in param_sets:
            mask = torch.rand(shape, dtype=torch.float32)
            ref = original_feather_reference(mask, l, t, r, b)
            res = FeatherMask.execute(mask, l, t, r, b)[0]

            assert res.shape == ref.shape
            assert torch.allclose(res, ref, atol=1e-6), f"Mismatch for shape={shape}, params=({l},{t},{r},{b})"

def test_feather_mask_clamping():
    mask = torch.ones((1, 20, 20), dtype=torch.float32)
    # Pass feather parameters exceeding mask resolution
    res = FeatherMask.execute(mask, 50, 50, 50, 50)[0]
    ref = original_feather_reference(mask, 50, 50, 50, 50)
    assert torch.allclose(res, ref, atol=1e-6)
