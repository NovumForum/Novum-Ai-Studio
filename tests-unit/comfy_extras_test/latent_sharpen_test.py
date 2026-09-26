import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn.functional as F

mock_device = torch.device("cpu")
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_mm = MagicMock()
mock_mm.get_torch_device.return_value = mock_device
mock_mm.intermediate_device.return_value = mock_device

with patch.dict('sys.modules', {
    'nodes': mock_nodes,
    'comfy.model_management': mock_mm,
}):
    import comfy
    comfy.model_management = mock_mm
    from comfy_extras.nodes_post_processing import Blur, Sharpen, gaussian_kernel_1d, gaussian_blur_1d
    from comfy_extras.nodes_latent import LatentOperationSharpen


class TestBlurSharpenSeparable(unittest.TestCase):
    def test_gaussian_kernel_1d_sum(self):
        kernel = gaussian_kernel_1d(15, 2.0)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=6)

    def test_blur_equivalence(self):
        generator = torch.Generator().manual_seed(42)
        # Random image batch [B, H, W, C]
        image = torch.randn(2, 64, 64, 3, generator=generator)
        blur_radius = 5
        sigma = 1.5
        kernel_size = blur_radius * 2 + 1

        # Calculate reference 2D Gaussian blur
        def reference_2d_blur(img, radius, sig):
            channels = img.shape[-1]
            x, y = torch.meshgrid(
                torch.linspace(-1, 1, kernel_size),
                torch.linspace(-1, 1, kernel_size),
                indexing="ij"
            )
            d = torch.sqrt(x * x + y * y)
            g = torch.exp(-(d * d) / (2.0 * sig * sig))
            kernel_2d = (g / g.sum()).repeat(channels, 1, 1).unsqueeze(1)

            tensor_img = img.permute(0, 3, 1, 2)
            padded = F.pad(tensor_img, (radius, radius, radius, radius), 'reflect')
            blurred_2d = F.conv2d(padded, kernel_2d, padding=kernel_size // 2, groups=channels)[:, :, radius:-radius, radius:-radius]
            return blurred_2d.permute(0, 2, 3, 1)

        ref_out = reference_2d_blur(image, blur_radius, sigma)
        node_out = Blur.execute(image, blur_radius=blur_radius, sigma=sigma).args[0]

        max_diff = torch.max(torch.abs(ref_out - node_out)).item()
        self.assertLess(max_diff, 1e-5)

    def test_blur_zero_radius(self):
        generator = torch.Generator().manual_seed(42)
        image = torch.randn(1, 32, 32, 3, generator=generator)
        node_out = Blur.execute(image, blur_radius=0, sigma=1.0).args[0]
        self.assertTrue(torch.equal(image, node_out))

    def test_sharpen_equivalence(self):
        generator = torch.Generator().manual_seed(42)
        image = torch.rand(2, 64, 64, 3, generator=generator)
        sharpen_radius = 4
        sigma = 1.2
        alpha = 0.5
        kernel_size = sharpen_radius * 2 + 1

        # Calculate reference 2D Sharpen
        def reference_2d_sharpen(img, radius, sig, alp):
            channels = img.shape[-1]
            x, y = torch.meshgrid(
                torch.linspace(-1, 1, kernel_size),
                torch.linspace(-1, 1, kernel_size),
                indexing="ij"
            )
            d = torch.sqrt(x * x + y * y)
            g = torch.exp(-(d * d) / (2.0 * sig * sig))
            g2d = g / g.sum()

            kernel = g2d * -(alp * 10)
            center = kernel_size // 2
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
            kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

            tensor_img = img.permute(0, 3, 1, 2)
            padded = F.pad(tensor_img, (radius, radius, radius, radius), 'reflect')
            sharpened = F.conv2d(padded, kernel, padding=center, groups=channels)[:, :, radius:-radius, radius:-radius]
            return torch.clamp(sharpened.permute(0, 2, 3, 1), 0, 1)

        ref_out = reference_2d_sharpen(image, sharpen_radius, sigma, alpha)
        node_out = Sharpen.execute(image, sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha).args[0]

        max_diff = torch.max(torch.abs(ref_out - node_out)).item()
        self.assertLess(max_diff, 1e-4)

    def test_latent_operation_sharpen_equivalence(self):
        generator = torch.Generator().manual_seed(42)
        latent = torch.randn(2, 4, 32, 32, generator=generator)
        sharpen_radius = 3
        sigma = 1.0
        alpha = 0.2

        op = LatentOperationSharpen.execute(sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha).args[0]
        node_out = op(latent=latent)

        # Check zero radius returns original latent
        op_zero = LatentOperationSharpen.execute(sharpen_radius=0, sigma=sigma, alpha=alpha).args[0]
        zero_out = op_zero(latent=latent)
        self.assertTrue(torch.equal(latent, zero_out))

        # Output shape check
        self.assertEqual(node_out.shape, latent.shape)


if __name__ == "__main__":
    unittest.main()
