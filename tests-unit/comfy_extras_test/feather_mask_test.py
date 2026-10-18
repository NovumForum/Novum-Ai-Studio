import torch
from comfy_extras.nodes_mask import FeatherMask

def ref_feather(mask, left, top, right, bottom):
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

def test_feather_mask_exact_match():
    torch.manual_seed(42)
    shapes = [(1, 64, 64), (2, 128, 128), (4, 256, 512)]
    margins = [
        (0, 0, 0, 0),
        (10, 10, 10, 10),
        (5, 0, 15, 20),
        (100, 100, 100, 100),
    ]

    for shape in shapes:
        for left, top, right, bottom in margins:
            mask = torch.rand(shape, dtype=torch.float32)
            res_node = FeatherMask.execute(mask, left, top, right, bottom)[0]
            res_ref = ref_feather(mask, left, top, right, bottom)

            assert res_node.shape == res_ref.shape
            torch.testing.assert_close(res_node, res_ref, rtol=1e-5, atol=1e-5)
