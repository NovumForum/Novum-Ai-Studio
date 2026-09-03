import torch
import torch.nn.functional as F
import comfy.cli_args

comfy.cli_args.args.cpu = True

from comfy_extras.nodes_post_processing import Blur, Sharpen, gaussian_kernel

def legacy_gaussian_blur(image: torch.Tensor, blur_radius: int, sigma: float) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    kernel_size = blur_radius * 2 + 1
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

    permuted = image.permute(0, 3, 1, 2)
    padded = F.pad(permuted, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
    blurred = F.conv2d(padded, kernel, padding=kernel_size // 2, groups=channels)[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]
    return blurred.permute(0, 2, 3, 1)

def legacy_gaussian_sharpen(image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    kernel_size = sharpen_radius * 2 + 1
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device) * -(alpha * 10)
    kernel = kernel.to(dtype=image.dtype)
    center = kernel_size // 2
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
    kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

    permuted = image.permute(0, 3, 1, 2)
    padded = F.pad(permuted, (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius), 'reflect')
    sharpened = F.conv2d(padded, kernel, padding=center, groups=channels)[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
    return torch.clamp(sharpened.permute(0, 2, 3, 1), 0.0, 1.0)


def test_blur_zero_radius():
    image = torch.rand(2, 64, 64, 3)
    out = Blur.execute(image, blur_radius=0, sigma=1.0).args[0]
    assert torch.equal(out, image)


def test_blur_shape_and_range():
    image = torch.rand(2, 64, 64, 3)
    out = Blur.execute(image, blur_radius=5, sigma=1.5).args[0]
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_blur_rgba_channels():
    image = torch.rand(1, 32, 32, 4)
    out = Blur.execute(image, blur_radius=3, sigma=1.0).args[0]
    assert out.shape == (1, 32, 32, 4)


def test_blur_equivalence_to_2d():
    image = torch.rand(2, 64, 64, 3)
    for radius in [1, 3, 7, 15]:
        out_opt = Blur.execute(image, blur_radius=radius, sigma=2.0).args[0]
        out_legacy = legacy_gaussian_blur(image, blur_radius=radius, sigma=2.0)
        max_diff = torch.max(torch.abs(out_opt - out_legacy)).item()
        assert max_diff < 1e-5, f"Blur radius {radius} max difference {max_diff} exceeded tolerance"


def test_sharpen_zero_radius_or_alpha():
    image = torch.rand(2, 64, 64, 3)
    out1 = Sharpen.execute(image, sharpen_radius=0, sigma=1.0, alpha=1.0).args[0]
    assert torch.equal(out1, image)

    out2 = Sharpen.execute(image, sharpen_radius=3, sigma=1.0, alpha=0.0).args[0]
    assert torch.equal(out2, image)


def test_sharpen_shape_and_range():
    image = torch.rand(2, 64, 64, 3)
    out = Sharpen.execute(image, sharpen_radius=5, sigma=1.5, alpha=1.2).args[0]
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_sharpen_equivalence_to_2d():
    image = torch.rand(2, 64, 64, 3)
    for radius in [1, 3, 7, 15]:
        out_opt = Sharpen.execute(image, sharpen_radius=radius, sigma=2.0, alpha=1.5).args[0]
        out_legacy = legacy_gaussian_sharpen(image, sharpen_radius=radius, sigma=2.0, alpha=1.5)
        max_diff = torch.max(torch.abs(out_opt - out_legacy)).item()
        assert max_diff < 1e-5, f"Sharpen radius {radius} max difference {max_diff} exceeded tolerance"
