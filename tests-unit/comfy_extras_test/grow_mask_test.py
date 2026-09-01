import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
import scipy.ndimage
import numpy as np
from comfy_extras.nodes_mask import GrowMask


def legacy_grow_mask(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def test_grow_mask_zero_expand():
    mask = torch.rand(2, 64, 64, dtype=torch.float32)
    res = GrowMask.execute(mask, expand=0, tapered_corners=True).args[0]
    assert res.shape == mask.shape
    torch.testing.assert_close(res, mask)


def test_grow_mask_positive_expand_tapered():
    mask = torch.rand(4, 128, 128, dtype=torch.float32)
    res = GrowMask.execute(mask, expand=5, tapered_corners=True).args[0]
    legacy = legacy_grow_mask(mask, expand=5, tapered_corners=True)
    assert res.shape == legacy.shape
    torch.testing.assert_close(res, legacy)


def test_grow_mask_positive_expand_square():
    mask = torch.rand(4, 128, 128, dtype=torch.float32)
    res = GrowMask.execute(mask, expand=5, tapered_corners=False).args[0]
    legacy = legacy_grow_mask(mask, expand=5, tapered_corners=False)
    assert res.shape == legacy.shape
    torch.testing.assert_close(res, legacy)


def test_grow_mask_negative_expand_tapered():
    mask = torch.rand(4, 128, 128, dtype=torch.float32)
    res = GrowMask.execute(mask, expand=-5, tapered_corners=True).args[0]
    legacy = legacy_grow_mask(mask, expand=-5, tapered_corners=True)
    assert res.shape == legacy.shape
    torch.testing.assert_close(res, legacy)


def test_grow_mask_negative_expand_square():
    mask = torch.rand(4, 128, 128, dtype=torch.float32)
    res = GrowMask.execute(mask, expand=-5, tapered_corners=False).args[0]
    legacy = legacy_grow_mask(mask, expand=-5, tapered_corners=False)
    assert res.shape == legacy.shape
    torch.testing.assert_close(res, legacy)
