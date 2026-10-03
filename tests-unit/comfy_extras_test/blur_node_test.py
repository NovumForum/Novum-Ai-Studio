from unittest.mock import patch, MagicMock
import torch
import pytest

# Mock nodes module and comfy_aimdo module to prevent C extension load failure
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_aimdo = MagicMock()

# Setup CLI args for CPU mode
import comfy.cli_args
comfy.cli_args.args = comfy.cli_args.parser.parse_args(["--cpu"])

# Patch modules before importing comfy_extras.nodes_post_processing
with patch.dict('sys.modules', {
    'nodes': mock_nodes,
    'comfy_aimdo': mock_aimdo,
    'comfy_aimdo.torch': mock_aimdo,
    'comfy_aimdo.model_vbar': mock_aimdo,
    'comfy_aimdo.control': mock_aimdo,
}):
    from comfy_extras.nodes_post_processing import Blur, gaussian_kernel_1d, gaussian_kernel

def reference_blur_2d(image: torch.Tensor, blur_radius: int, sigma: float) -> torch.Tensor:
    """Unoptimized 2D Gaussian convolution reference implementation."""
    import torch.nn.functional as F
    batch_size, height, width, channels = image.shape
    kernel_size = blur_radius * 2 + 1
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

    img_perm = image.permute(0, 3, 1, 2)
    padded_image = F.pad(img_perm, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
    blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]
    return blurred.permute(0, 2, 3, 1)

def test_blur_zero_radius():
    image = torch.rand((2, 64, 64, 3))
    out = Blur.execute(image, blur_radius=0, sigma=1.0)
    assert torch.equal(out.args[0], image)

def test_gaussian_kernel_1d_sum():
    for kernel_size in [3, 7, 15, 31]:
        g = gaussian_kernel_1d(kernel_size, 1.5)
        assert pytest.approx(g.sum().item(), abs=1e-6) == 1.0

@pytest.mark.parametrize("blur_radius", [1, 5, 15])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_blur_separable_matches_2d(blur_radius, channels):
    torch.manual_seed(42)
    image = torch.rand((2, 64, 64, channels), dtype=torch.float32)

    out_node = Blur.execute(image, blur_radius=blur_radius, sigma=2.0).args[0]
    out_ref = reference_blur_2d(image, blur_radius=blur_radius, sigma=2.0)

    assert out_node.shape == out_ref.shape
    # Ensure 1D separable convolution matches 2D convolution within floating point tolerance
    diff = (out_node - out_ref).abs().max().item()
    assert diff < 1e-5, f"Difference {diff} exceeded tolerance for radius {blur_radius}"
