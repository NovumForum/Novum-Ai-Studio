import sys
import types
import unittest.mock as mock
import pytest
import torch
import numpy as np
from PIL import Image

comfy_pkg = types.ModuleType('comfy')
comfy_pkg.model_management = mock.MagicMock()
comfy_pkg.model_management.get_torch_device.return_value = 'cpu'
comfy_pkg.model_management.intermediate_device.return_value = 'cpu'
comfy_pkg.utils = mock.MagicMock()
comfy_pkg.diffusers_load = mock.MagicMock()
comfy_pkg.sd = mock.MagicMock()

comfy_cli = types.ModuleType('comfy.cli_args')
comfy_cli.args = mock.MagicMock()
comfy_cli.args.cpu = True

mock_nodes = mock.MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_modules = {
    'comfy': comfy_pkg,
    'comfy.model_management': comfy_pkg.model_management,
    'comfy.utils': comfy_pkg.utils,
    'comfy.diffusers_load': comfy_pkg.diffusers_load,
    'comfy.sd': comfy_pkg.sd,
    'comfy.cli_args': comfy_cli,
    'nodes': mock_nodes
}

sys.modules.update(mock_modules)

from comfy_extras.nodes_post_processing import Quantize


def test_quantize_execute_none_dither():
    image = torch.rand((2, 64, 64, 3), dtype=torch.float32)
    output = Quantize.execute(image, colors=16, dither="none")
    assert output is not None
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape == image.shape
    assert output[0].dtype == torch.float32


def test_quantize_execute_floyd_steinberg():
    image = torch.rand((1, 32, 32, 3), dtype=torch.float32)
    output = Quantize.execute(image, colors=8, dither="floyd-steinberg")
    assert output[0].shape == image.shape


@pytest.mark.parametrize("order", [2, 4, 8, 16])
def test_quantize_bayer_modes(order):
    image = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    dither_name = f"bayer-{order}"
    output = Quantize.execute(image, colors=32, dither=dither_name)
    assert output[0].shape == image.shape
    assert (output[0] >= 0.0).all() and (output[0] <= 1.0).all()


def test_quantize_bayer_direct_method():
    im = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    pal_im = im.quantize(colors=16)

    result_im = Quantize.bayer(im, pal_im, order=8)
    assert isinstance(result_im, Image.Image)
    assert result_im.size == im.size
