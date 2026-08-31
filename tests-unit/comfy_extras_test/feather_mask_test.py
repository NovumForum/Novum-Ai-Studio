import torch
from comfy_extras.nodes_mask import FeatherMask

def feather_legacy(mask, left, top, right, bottom):
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
    res = FeatherMask.execute(mask, 0, 0, 0, 0).args[0]
    assert torch.allclose(mask, res)


def test_feather_mask_equivalence_with_legacy():
    test_cases = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (10, 15, 20, 25),
        (100, 100, 100, 100),
    ]
    for shape in [(1, 128, 128), (2, 64, 64)]:
        mask = torch.rand(shape, dtype=torch.float32)
        for left, top, right, bottom in test_cases:
            legacy_res = feather_legacy(mask, left, top, right, bottom)
            vec_res = FeatherMask.execute(mask, left, top, right, bottom).args[0]
            assert torch.allclose(legacy_res, vec_res, atol=1e-6)


def test_feather_mask_clamped_bounds():
    mask = torch.ones((1, 32, 32), dtype=torch.float32)
    # left/top/right/bottom larger than H or W
    legacy_res = feather_legacy(mask, 50, 50, 50, 50)
    vec_res = FeatherMask.execute(mask, 50, 50, 50, 50).args[0]
    assert torch.allclose(legacy_res, vec_res, atol=1e-6)
