import torch
import numpy as np
import scipy.ndimage
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_mask import GrowMask, ImageColorToMask


def scipy_reference_grow(mask: torch.Tensor, expand: int, tapered_corners: bool) -> torch.Tensor:
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask_r = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask_r:
        output = m.cpu().numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def test_grow_mask_mathematical_equivalence():
    mask = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]).unsqueeze(0)

    for expand in [0, 1, 2, -1, -2]:
        for tapered_corners in [True, False]:
            ref_output = scipy_reference_grow(mask, expand, tapered_corners)
            node_output = GrowMask.execute(mask, expand, tapered_corners)[0]

            assert node_output.shape == ref_output.shape
            diff = (node_output - ref_output).abs().max().item()
            assert diff == 0.0, f"Mismatch for expand={expand}, tapered_corners={tapered_corners}: max_diff={diff}"


def test_grow_mask_zero_expand():
    mask = torch.ones(2, 8, 8)
    output = GrowMask.execute(mask, 0, True)[0]
    assert torch.equal(output, mask.reshape((-1, 8, 8)))


def test_image_color_to_mask():
    image = torch.zeros(2, 4, 4, 3)
    image[0, 1, 1] = torch.tensor([1.0, 128.0 / 255.0, 64.0 / 255.0])
    color = 0xFF8040

    node_output = ImageColorToMask.execute(image, color)[0]

    assert node_output.shape == (2, 4, 4)
    assert node_output[0, 1, 1] == 1.0
    assert (node_output.sum().item()) == 1.0
