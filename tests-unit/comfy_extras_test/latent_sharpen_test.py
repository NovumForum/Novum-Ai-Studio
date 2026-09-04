import sys
from unittest.mock import MagicMock

# Mock comfy_aimdo module hierarchy before imports
comfy_aimdo = MagicMock()
sys.modules["comfy_aimdo"] = comfy_aimdo
sys.modules["comfy_aimdo.control"] = comfy_aimdo.control
sys.modules["comfy_aimdo.model_vbar"] = comfy_aimdo.model_vbar
sys.modules["comfy_aimdo.torch"] = comfy_aimdo.torch

import comfy.cli_args

comfy.cli_args.args.cpu = True

import pytest
import torch
import torch.nn.functional as F
import comfy_extras.nodes_post_processing
from comfy_extras.nodes_latent import LatentOperationSharpen


def legacy_sharpen(latent, sharpen_radius, sigma, alpha):
    luminance = (torch.linalg.vector_norm(latent, dim=(1)) + 1e-6)[:, None]
    normalized_latent = latent / luminance
    channels = latent.shape[1]

    kernel_size = sharpen_radius * 2 + 1
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    center = kernel_size // 2

    kernel *= alpha * -10
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

    padded_image = F.pad(
        normalized_latent,
        (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
        "reflect",
    )
    sharpened = F.conv2d(
        padded_image,
        kernel.repeat(channels, 1, 1).unsqueeze(1),
        padding=kernel_size // 2,
        groups=channels,
    )[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]

    return luminance * sharpened


def test_latent_operation_sharpen_zero_alpha():
    latent = torch.rand(2, 4, 32, 32)
    operation_node = LatentOperationSharpen()
    op_output = operation_node.execute(sharpen_radius=9, sigma=1.0, alpha=0.0)
    sharpen_fn = op_output.args[0]

    result = sharpen_fn(latent)
    assert result.shape == latent.shape
    assert torch.allclose(result, latent, atol=1e-5)


def test_latent_operation_sharpen_equivalence_with_legacy():
    torch.manual_seed(42)
    latent = torch.rand(2, 4, 64, 64)

    test_configs = [
        (1, 0.5, 0.1),
        (5, 1.5, 0.2),
        (9, 2.0, 0.5),
        (15, 3.0, 1.0),
    ]

    for radius, sigma, alpha in test_configs:
        operation_node = LatentOperationSharpen()
        op_output = operation_node.execute(
            sharpen_radius=radius, sigma=sigma, alpha=alpha
        )
        sharpen_fn = op_output.args[0]

        opt_result = sharpen_fn(latent)
        leg_result = legacy_sharpen(latent, radius, sigma, alpha)

        assert opt_result.shape == latent.shape
        assert torch.allclose(opt_result, leg_result, atol=1e-4)


def test_latent_operation_sharpen_shapes_and_dtypes():
    for batch_size, channels in [(1, 4), (3, 8), (2, 16)]:
        latent = torch.randn(batch_size, channels, 40, 40, dtype=torch.float32)
        operation_node = LatentOperationSharpen()
        op_output = operation_node.execute(sharpen_radius=3, sigma=1.0, alpha=0.1)
        sharpen_fn = op_output.args[0]

        result = sharpen_fn(latent)
        assert result.shape == (batch_size, channels, 40, 40)
        assert result.dtype == torch.float32
