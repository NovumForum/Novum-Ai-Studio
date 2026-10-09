import comfy.cli_args
comfy.cli_args.args.cpu = True

import pytest
import torch
import kornia
from comfy_extras.nodes_morphology import ImageRGBToYUV, ImageYUVToRGB


def reference_yuv_to_rgb(Y, U, V):
    image = torch.cat([torch.mean(Y, dim=-1, keepdim=True), torch.mean(U, dim=-1, keepdim=True), torch.mean(V, dim=-1, keepdim=True)], dim=-1)
    out = kornia.color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1)
    return out


def test_image_rgb_to_yuv_shapes():
    batch_size, h, w, c = 2, 64, 64, 3
    rgb_image = torch.rand(batch_size, h, w, c)

    res = ImageRGBToYUV.execute(rgb_image)
    y, u, v = res.args

    assert y.shape == (batch_size, h, w, c)
    assert u.shape == (batch_size, h, w, c)
    assert v.shape == (batch_size, h, w, c)

    # Check that channels are duplicated as expected by expand_as
    assert torch.allclose(y[..., 0], y[..., 1])
    assert torch.allclose(y[..., 0], y[..., 2])
    assert torch.allclose(u[..., 0], u[..., 1])
    assert torch.allclose(u[..., 0], u[..., 2])
    assert torch.allclose(v[..., 0], v[..., 1])
    assert torch.allclose(v[..., 0], v[..., 2])


def test_image_yuv_to_rgb_equivalence():
    batch_size, h, w, c = 2, 64, 64, 3
    rgb_image = torch.rand(batch_size, h, w, c)

    res_yuv = ImageRGBToYUV.execute(rgb_image)
    y, u, v = res_yuv.args

    ref_out = reference_yuv_to_rgb(y, u, v)
    opt_out = ImageYUVToRGB.execute(y, u, v).args[0]

    assert opt_out.shape == (batch_size, h, w, c)
    assert torch.allclose(ref_out, opt_out, atol=1e-5)


def test_image_yuv_to_rgb_roundtrip():
    batch_size, h, w, c = 2, 32, 32, 3
    rgb_image = torch.rand(batch_size, h, w, c)

    res_yuv = ImageRGBToYUV.execute(rgb_image)
    y, u, v = res_yuv.args

    res_rgb = ImageYUVToRGB.execute(y, u, v).args[0]

    assert res_rgb.shape == rgb_image.shape
    assert torch.allclose(rgb_image, res_rgb, atol=1e-3)


def test_image_yuv_to_rgb_single_channel_inputs():
    batch_size, h, w = 2, 32, 32
    y_1c = torch.rand(batch_size, h, w, 1)
    u_1c = torch.rand(batch_size, h, w, 1)
    v_1c = torch.rand(batch_size, h, w, 1)

    opt_out = ImageYUVToRGB.execute(y_1c, u_1c, v_1c).args[0]
    ref_out = reference_yuv_to_rgb(y_1c, u_1c, v_1c)

    assert opt_out.shape == (batch_size, h, w, 3)
    assert torch.allclose(ref_out, opt_out, atol=1e-5)
