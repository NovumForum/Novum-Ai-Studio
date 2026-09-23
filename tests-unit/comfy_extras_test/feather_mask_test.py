import unittest.mock as mock
import torch

# Mock torch.cuda for CPU environments before importing comfy / nodes modules
torch.cuda.is_available = mock.MagicMock(return_value=False)
torch.cuda.current_device = mock.MagicMock(return_value='cpu')
torch.cuda.get_device_properties = mock.MagicMock(return_value=mock.MagicMock(total_memory=8 * 1024 ** 3))
torch.cuda.memory_stats = mock.MagicMock(return_value={})
torch.cuda.mem_get_info = mock.MagicMock(return_value=(8 * 1024 ** 3, 8 * 1024 ** 3))

from comfy_extras.nodes_mask import FeatherMask

def feather_reference(mask: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
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
    mask = torch.ones((1, 64, 64))
    res = FeatherMask.execute(mask, 0, 0, 0, 0)
    out = res.value if hasattr(res, 'value') else res[0]
    assert torch.allclose(out, mask)

def test_feather_mask_equivalence():
    shapes = [(1, 32, 32), (4, 100, 100), (2, 10, 20)]
    param_combinations = [
        (5, 5, 5, 5),
        (0, 10, 0, 10),
        (15, 0, 15, 0),
        (50, 50, 50, 50),  # clamped out of bounds
        (1, 1, 1, 1),
    ]

    for shape in shapes:
        mask = torch.rand(shape)
        for left, top, right, bottom in param_combinations:
            expected = feather_reference(mask, left, top, right, bottom)
            res = FeatherMask.execute(mask, left, top, right, bottom)
            actual = res.value if hasattr(res, 'value') else res[0]
            assert torch.allclose(actual, expected, atol=1e-6), f"Mismatch for shape {shape} with params ({left}, {top}, {right}, {bottom})"
