import torch
import torch.nn.functional as F
from comfy_extras.nodes_post_processing import Blur, gaussian_kernel


# Reference 2D Gaussian Blur implementation to compare against
def reference_blur_2d(image: torch.Tensor, blur_radius: int, sigma: float) -> torch.Tensor:
    if blur_radius == 0:
        return image

    batch_size, height, width, channels = image.shape
    kernel_size = blur_radius * 2 + 1

    # Replicate the original 2D kernel logic
    kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

    image_t = image.permute(0, 3, 1, 2)  # Torch wants (B, C, H, W)
    padded_image = F.pad(image_t, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')

    # Original double-padding logic
    blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius]
    blurred = blurred.permute(0, 2, 3, 1)

    return blurred


class TestImageBlur:

    def create_test_image(self, batch_size=1, height=64, width=64, channels=3):
        """Helper to create test images with specific dimensions"""
        return torch.rand(batch_size, height, width, channels)

    def test_blur_radius_0_passthrough(self):
        """Test that when blur_radius is 0, the image is returned unchanged"""
        node = Blur()
        image = self.create_test_image()

        result = node.execute(image, blur_radius=0, sigma=1.0)

        assert len(result.result) == 1
        assert torch.equal(result[0], image)

    def test_blur_output_shape(self):
        """Test that output image has the exact same shape as input"""
        node = Blur()
        image = self.create_test_image(batch_size=2, height=32, width=48, channels=3)

        result = node.execute(image, blur_radius=2, sigma=1.5)

        assert len(result.result) == 1
        assert result[0].shape == image.shape

    def test_blur_mathematical_equivalence(self):
        """Test that our 1D separable implementation matches the original 2D implementation"""
        node = Blur()

        # Test various parameters
        test_cases = [
            {"batch_size": 1, "height": 32, "width": 32, "channels": 3, "blur_radius": 1, "sigma": 0.5},
            {"batch_size": 2, "height": 48, "width": 48, "channels": 3, "blur_radius": 3, "sigma": 1.0},
            {"batch_size": 1, "height": 64, "width": 64, "channels": 1, "blur_radius": 5, "sigma": 2.0},
            {"batch_size": 1, "height": 64, "width": 64, "channels": 3, "blur_radius": 8, "sigma": 3.0},
        ]

        for case in test_cases:
            image = self.create_test_image(
                batch_size=case["batch_size"],
                height=case["height"],
                width=case["width"],
                channels=case["channels"]
            )

            ref_out = reference_blur_2d(image, case["blur_radius"], case["sigma"])
            opt_out = node.execute(image, case["blur_radius"], case["sigma"])[0]

            # Assert they are mathematically equivalent (within floating point precision limits)
            diff = (ref_out - opt_out).abs()
            assert diff.max().item() < 1e-5
            assert diff.mean().item() < 1e-6

    def test_blur_large_radius(self):
        """Test execution with larger blur radius to ensure no crash/overflow"""
        node = Blur()
        image = self.create_test_image(batch_size=1, height=128, width=128, channels=3)

        result = node.execute(image, blur_radius=15, sigma=5.0)

        assert len(result.result) == 1
        assert result[0].shape == image.shape
