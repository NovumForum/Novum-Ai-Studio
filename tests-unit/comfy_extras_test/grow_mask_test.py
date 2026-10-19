import sys
import torch
import numpy as np
import scipy.ndimage
from unittest.mock import MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

with MagicMock() as mock_context:
    sys.modules['nodes'] = mock_nodes
    from comfy_extras.nodes_mask import GrowMask


def reference_grow_mask(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask_reshaped = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask_reshaped:
        output = m.cpu().numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def test_grow_mask_zero_expand():
    mask = torch.rand((2, 64, 64), dtype=torch.float32)
    output = GrowMask.execute(mask, expand=0, tapered_corners=True)[0]
    assert torch.equal(output, mask)


def test_grow_mask_dilation_untapered():
    torch.manual_seed(42)
    mask = (torch.rand((2, 32, 32)) > 0.7).float()
    for expand in [1, 3, 5]:
        ref = reference_grow_mask(mask, expand, tapered_corners=False)
        res = GrowMask.execute(mask, expand=expand, tapered_corners=False)[0]
        assert torch.equal(res, ref), f"Mismatch for expand={expand}, tapered=False"


def test_grow_mask_dilation_tapered():
    torch.manual_seed(42)
    mask = (torch.rand((2, 32, 32)) > 0.7).float()
    for expand in [1, 3, 5]:
        ref = reference_grow_mask(mask, expand, tapered_corners=True)
        res = GrowMask.execute(mask, expand=expand, tapered_corners=True)[0]
        assert torch.equal(res, ref), f"Mismatch for expand={expand}, tapered=True"


def test_grow_mask_erosion_untapered():
    torch.manual_seed(42)
    mask = (torch.rand((2, 32, 32)) > 0.3).float()
    for expand in [-1, -3, -5]:
        ref = reference_grow_mask(mask, expand, tapered_corners=False)
        res = GrowMask.execute(mask, expand=expand, tapered_corners=False)[0]
        assert torch.equal(res, ref), f"Mismatch for expand={expand}, tapered=False"


def test_grow_mask_erosion_tapered():
    torch.manual_seed(42)
    mask = (torch.rand((2, 32, 32)) > 0.3).float()
    for expand in [-1, -3, -5]:
        ref = reference_grow_mask(mask, expand, tapered_corners=True)
        res = GrowMask.execute(mask, expand=expand, tapered_corners=True)[0]
        assert torch.equal(res, ref), f"Mismatch for expand={expand}, tapered=True"


def test_grow_mask_batch_shapes():
    mask_2d = (torch.rand((32, 32)) > 0.5).float()
    res_2d = GrowMask.execute(mask_2d, expand=2, tapered_corners=True)[0]
    assert res_2d.shape == (1, 32, 32)

    mask_3d = (torch.rand((4, 32, 32)) > 0.5).float()
    res_3d = GrowMask.execute(mask_3d, expand=2, tapered_corners=True)[0]
    assert res_3d.shape == (4, 32, 32)
