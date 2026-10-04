import comfy.cli_args
comfy.cli_args.args.cpu = True

import pytest
import torch
import torch.nn.functional as F

from comfy_extras.nodes_post_processing import Blur, Sharpen, gaussian_kernel


def gaussian_kernel_2d_reference(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, kernel_size),
        torch.linspace(-1, 1, kernel_size),
        indexing="ij",
    )
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()


def blur_reference(image: torch.Tensor, blur_radius: int, sigma: float):
    if blur_radius == 0:
        return image
    batch_size, height, width, channels = image.shape
    kernel_size = blur_radius * 2 + 1
    kernel = gaussian_kernel_2d_reference(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)
    tensor_image = image.permute(0, 3, 1, 2)
    padded_image = F.pad(
        tensor_image, (blur_radius, blur_radius, blur_radius, blur_radius), "reflect"
    )
    blurred = F.conv2d(
        padded_image, kernel, padding=kernel_size // 2, groups=channels
    )[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]
    return blurred.permute(0, 2, 3, 1)


def sharpen_reference(
    image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float
):
    if sharpen_radius == 0:
        return image
    batch_size, height, width, channels = image.shape
    kernel_size = sharpen_radius * 2 + 1
    kernel = gaussian_kernel_2d_reference(kernel_size, sigma) * -(alpha * 10)
    kernel = kernel.to(dtype=image.dtype)
    center = kernel_size // 2
    kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
    kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

    tensor_image = image.permute(0, 3, 1, 2)
    tensor_image = F.pad(
        tensor_image,
        (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
        "reflect",
    )
    sharpened = F.conv2d(
        tensor_image, kernel, padding=center, groups=channels
    )[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
    sharpened = sharpened.permute(0, 2, 3, 1)
    return torch.clamp(sharpened, 0, 1)


def test_blur_zero_radius():
    image = torch.rand(2, 32, 32, 3)
    res = Blur.execute(image, blur_radius=0, sigma=1.0)
    out = res.args[0]
    assert torch.equal(out, image)


@pytest.mark.parametrize("blur_radius", [1, 3, 5])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_blur_execution_and_numerical_equivalence(blur_radius: int, channels: int):
    torch.manual_seed(42)
    image = torch.rand(2, 64, 64, channels)
    sigma = 1.5

    res = Blur.execute(image, blur_radius=blur_radius, sigma=sigma)
    out = res.args[0]

    assert out.shape == image.shape
    ref = blur_reference(image, blur_radius=blur_radius, sigma=sigma)

    # Verify numerical equivalence within single-precision floating point tolerances
    assert torch.allclose(out, ref, atol=1e-5)


def test_sharpen_zero_radius():
    image = torch.rand(2, 32, 32, 3)
    res = Sharpen.execute(image, sharpen_radius=0, sigma=1.0, alpha=1.0)
    out = res.args[0]
    assert torch.equal(out, image)


@pytest.mark.parametrize("sharpen_radius", [1, 3, 5])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_sharpen_execution_and_numerical_equivalence(sharpen_radius: int, channels: int):
    torch.manual_seed(42)
    image = torch.rand(2, 64, 64, channels)
    sigma = 1.0
    alpha = 0.5

    res = Sharpen.execute(image, sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha)
    out = res.args[0]

    assert out.shape == image.shape
    ref = sharpen_reference(image, sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha)

    # Verify numerical equivalence within single-precision floating point tolerances
    assert torch.allclose(out, ref, atol=1e-5)
