import unittest.mock
import torch

# Mock CUDA functions for CPU-only test environment before importing comfy modules
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda dev=None: unittest.mock.MagicMock(total_memory=8 * 1024 * 1024 * 1024)
torch.cuda.mem_get_info = lambda dev=None: (4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
torch.cuda.memory_stats = lambda dev=None: {'reserved_bytes.all.current': 0}

import comfy.model_management
comfy.model_management.cpu_state = comfy.model_management.CPUState.CPU

import pytest
import torch.nn.functional as F
from comfy_extras.nodes_post_processing import Blur, Sharpen, gaussian_kernel
from comfy_extras.nodes_latent import LatentOperationSharpen

def reference_gaussian_blur_2d(image: torch.Tensor, blur_radius: int, sigma: float) -> torch.Tensor:
    if blur_radius == 0:
        return image
    batch_size, height, width, channels = image.shape
    kernel_size = blur_radius * 2 + 1
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

    img = image.permute(0, 3, 1, 2)
    padded_image = F.pad(img, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
    blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]
    return blurred.permute(0, 2, 3, 1)

def reference_sharpen_2d(image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float) -> torch.Tensor:
    if sharpen_radius == 0 or alpha == 0.0:
        return image
    batch_size, height, width, channels = image.shape
    kernel_size = sharpen_radius * 2 + 1
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device) * -(alpha * 10)
    kernel = kernel.to(dtype=image.dtype)
    center = kernel_size // 2
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
    kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

    tensor_image = image.permute(0, 3, 1, 2)
    tensor_image = F.pad(tensor_image, (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius), 'reflect')
    sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
    sharpened = sharpened.permute(0, 2, 3, 1)
    return torch.clamp(sharpened, 0, 1)

def test_blur_zero_radius():
    img = torch.rand(1, 64, 64, 3)
    out = Blur.execute(img, blur_radius=0, sigma=1.0)
    assert torch.equal(out[0], img)

def test_blur_numerical_equivalence():
    torch.manual_seed(42)
    img = torch.rand(2, 128, 128, 3)
    for radius in [1, 3, 7, 15]:
        out_opt = Blur.execute(img, blur_radius=radius, sigma=2.0)[0]
        out_ref = reference_gaussian_blur_2d(img, blur_radius=radius, sigma=2.0)
        max_diff = (out_opt - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"Blur radius {radius} max_diff {max_diff} exceeds tolerance"

def test_sharpen_zero_alpha_or_radius():
    img = torch.rand(1, 64, 64, 3)
    out_r0 = Sharpen.execute(img, sharpen_radius=0, sigma=1.0, alpha=1.0)[0]
    assert torch.equal(out_r0, img)

    out_a0 = Sharpen.execute(img, sharpen_radius=5, sigma=1.0, alpha=0.0)[0]
    assert torch.equal(out_a0, img)

def test_sharpen_numerical_equivalence():
    torch.manual_seed(42)
    img = torch.rand(2, 128, 128, 3)
    for radius in [1, 3, 5]:
        out_opt = Sharpen.execute(img, sharpen_radius=radius, sigma=1.5, alpha=0.8)[0]
        out_ref = reference_sharpen_2d(img, sharpen_radius=radius, sigma=1.5, alpha=0.8)
        max_diff = (out_opt - out_ref).abs().max().item()
        assert max_diff < 1e-5, f"Sharpen radius {radius} max_diff {max_diff} exceeds tolerance"

def test_latent_operation_sharpen():
    torch.manual_seed(42)
    latent = torch.randn(2, 4, 32, 32)
    op = LatentOperationSharpen.execute(sharpen_radius=9, sigma=1.0, alpha=0.1)[0]
    out_latent = op(latent)

    assert out_latent.shape == latent.shape
    assert not torch.isnan(out_latent).any()

    # Test early return for zero radius or alpha
    op_r0 = LatentOperationSharpen.execute(sharpen_radius=0, sigma=1.0, alpha=0.1)[0]
    assert torch.equal(op_r0(latent), latent)

    op_a0 = LatentOperationSharpen.execute(sharpen_radius=9, sigma=1.0, alpha=0.0)[0]
    assert torch.equal(op_a0(latent), latent)
