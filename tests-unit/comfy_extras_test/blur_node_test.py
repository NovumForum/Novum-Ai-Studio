import pytest
import torch
from unittest.mock import MagicMock, patch

# Safely mock torch.cuda before importing comfy.model_management to prevent CUDA initialization errors on CPU-only runners
with patch('torch.cuda.is_available', return_value=False), \
     patch('torch.cuda.current_device', return_value=0), \
     patch('torch.cuda.memory_stats', return_value={'reserved_bytes.all.current': 0}), \
     patch('torch.cuda.mem_get_info', return_value=(0, 1024 * 1024 * 1024)):
    import comfy.model_management
    comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    comfy.model_management.intermediate_device = lambda: torch.device("cpu")
    from comfy_extras.nodes_post_processing import Blur, gaussian_kernel_1d, gaussian_kernel

def test_gaussian_kernel_1d_normalization():
    k1d = gaussian_kernel_1d(15, 2.0)
    assert k1d.ndim == 1
    assert len(k1d) == 15
    assert torch.isclose(k1d.sum(), torch.tensor(1.0))

def test_blur_zero_radius():
    img = torch.rand(1, 64, 64, 3)
    res = Blur.execute(img, blur_radius=0, sigma=1.0)
    assert torch.equal(res[0], img)

def test_blur_shape_and_dtype():
    img = torch.rand(2, 128, 128, 3, dtype=torch.float32)
    res = Blur.execute(img, blur_radius=5, sigma=1.5)
    out_tensor = res[0]
    assert out_tensor.shape == img.shape
    assert out_tensor.dtype == img.dtype

def test_blur_mathematical_equivalence():
    """Verify 1D separable convolution produces identical output to 2D convolution within precision limits."""
    img = torch.rand(1, 100, 100, 3)
    blur_radius = 10
    sigma = 2.5
    channels = img.shape[3]
    kernel_size = blur_radius * 2 + 1

    # 2D direct reference calculation
    k2d = gaussian_kernel(kernel_size, sigma, device=img.device).repeat(channels, 1, 1).unsqueeze(1)
    img_perm = img.permute(0, 3, 1, 2)
    padded = torch.nn.functional.pad(img_perm, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
    ref_2d = torch.nn.functional.conv2d(padded, k2d, padding=kernel_size // 2, groups=channels)[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius].permute(0, 2, 3, 1)

    # 1D separable Blur.execute result
    res_node = Blur.execute(img, blur_radius=blur_radius, sigma=sigma)[0]

    max_diff = (ref_2d - res_node).abs().max().item()
    assert max_diff < 1e-5
