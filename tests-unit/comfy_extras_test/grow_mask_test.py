from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np
import scipy.ndimage

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_mask import GrowMask


def scipy_grow_mask_reference(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    m_reshaped = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in m_reshaped:
        arr = m.cpu().numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
            else:
                arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
        out.append(torch.from_numpy(arr))
    stacked = torch.stack(out, dim=0)
    return stacked.reshape(mask.shape)


@pytest.mark.parametrize("expand", [0, 1, 5, -1, -5, 10, -10])
@pytest.mark.parametrize("tapered_corners", [True, False])
def test_grow_mask_scipy_equivalence(expand: int, tapered_corners: bool):
    gen = torch.Generator().manual_seed(42)
    mask = (torch.rand((2, 64, 64), generator=gen) > 0.5).float()

    expected = scipy_grow_mask_reference(mask, expand, tapered_corners)
    result = GrowMask.execute(mask, expand, tapered_corners)

    actual = result.result[0]

    assert actual.shape == mask.shape
    assert torch.equal(actual, expected)


def test_grow_mask_zero_expand():
    gen = torch.Generator().manual_seed(100)
    mask = torch.rand((3, 32, 32), generator=gen)
    result = GrowMask.execute(mask, expand=0, tapered_corners=True)
    actual = result.result[0]

    assert torch.equal(actual, mask)


@pytest.mark.parametrize("shape", [(64, 64), (1, 64, 64), (4, 64, 64)])
def test_grow_mask_various_shapes(shape):
    gen = torch.Generator().manual_seed(123)
    mask = torch.rand(shape, generator=gen)
    expected = scipy_grow_mask_reference(mask, expand=3, tapered_corners=False)
    result = GrowMask.execute(mask, expand=3, tapered_corners=False)
    actual = result.result[0]

    assert actual.shape == mask.shape
    assert torch.equal(actual, expected)
