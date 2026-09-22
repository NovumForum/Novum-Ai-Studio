from unittest.mock import patch, MagicMock
import numpy as np
import scipy.ndimage
import torch

# Mock torch.cuda and nodes before import
mock_cuda = MagicMock()
mock_cuda.is_available.return_value = False
mock_cuda.current_device.return_value = 0

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

with patch('torch.cuda', mock_cuda), patch.dict('sys.modules', {'nodes': mock_nodes}):
    from comfy_extras.nodes_mask import GrowMask


def grow_mask_scipy_reference(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
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
        out.append(torch.from_numpy(output))
    res = torch.stack(out, dim=0)
    return res.reshape(mask.shape)


def test_grow_mask_zero_expand():
    mask = torch.rand(2, 64, 64)
    res = GrowMask.execute(mask, expand=0, tapered_corners=True)[0]
    assert torch.equal(res, mask)


def test_grow_mask_positive_expand_tapered():
    torch.manual_seed(42)
    mask = (torch.rand(2, 64, 64) > 0.6).float()
    res_pytorch = GrowMask.execute(mask, expand=5, tapered_corners=True)[0]
    res_scipy = grow_mask_scipy_reference(mask, expand=5, tapered_corners=True)
    assert torch.allclose(res_pytorch, res_scipy, atol=1e-6)


def test_grow_mask_positive_expand_non_tapered():
    torch.manual_seed(42)
    mask = (torch.rand(2, 64, 64) > 0.6).float()
    res_pytorch = GrowMask.execute(mask, expand=5, tapered_corners=False)[0]
    res_scipy = grow_mask_scipy_reference(mask, expand=5, tapered_corners=False)
    assert torch.allclose(res_pytorch, res_scipy, atol=1e-6)


def test_grow_mask_negative_expand_tapered():
    torch.manual_seed(42)
    mask = (torch.rand(2, 64, 64) > 0.4).float()
    res_pytorch = GrowMask.execute(mask, expand=-5, tapered_corners=True)[0]
    res_scipy = grow_mask_scipy_reference(mask, expand=-5, tapered_corners=True)
    assert torch.allclose(res_pytorch, res_scipy, atol=1e-6)


def test_grow_mask_negative_expand_non_tapered():
    torch.manual_seed(42)
    mask = (torch.rand(2, 64, 64) > 0.4).float()
    res_pytorch = GrowMask.execute(mask, expand=-5, tapered_corners=False)[0]
    res_scipy = grow_mask_scipy_reference(mask, expand=-5, tapered_corners=False)
    assert torch.allclose(res_pytorch, res_scipy, atol=1e-6)


def test_grow_mask_continuous_values():
    torch.manual_seed(42)
    mask = torch.rand(1, 32, 32)
    res_pytorch = GrowMask.execute(mask, expand=3, tapered_corners=True)[0]
    res_scipy = grow_mask_scipy_reference(mask, expand=3, tapered_corners=True)
    assert torch.allclose(res_pytorch, res_scipy, atol=1e-6)
