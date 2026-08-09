import torch
import torch.nn.functional as F
from comfy_extras.nodes_post_processing import Blur, Sharpen, gaussian_kernel_1d, gaussian_kernel
from comfy_extras.nodes_latent import LatentOperationSharpen

class TestImageBlurAndSharpen:

    def create_test_image(self, batch_size=1, height=64, width=64, channels=3):
        return torch.rand(batch_size, height, width, channels)

    def test_gaussian_kernel_1d_sum(self):
        """Test that the generated 1D Gaussian kernel sums to 1.0."""
        kernel = gaussian_kernel_1d(15, 2.0)
        assert abs(kernel.sum().item() - 1.0) < 1e-6

    def test_blur_zero_or_negative_radius(self):
        """Test that Blur returns the input unchanged when radius is 0 or negative."""
        image = self.create_test_image()
        blur_node = Blur()

        # Zero radius
        out_zero = blur_node.execute(image, blur_radius=0, sigma=1.0)
        assert torch.equal(out_zero[0], image)

        # Negative radius
        out_neg = blur_node.execute(image, blur_radius=-5, sigma=1.0)
        assert torch.equal(out_neg[0], image)

    def test_blur_shape_preservation(self):
        """Test that Blur preserves the spatial and batch dimensions of the input image."""
        image = self.create_test_image(batch_size=2, height=48, width=54, channels=4)
        blur_node = Blur()

        out = blur_node.execute(image, blur_radius=3, sigma=1.5)
        assert out[0].shape == image.shape

    def test_blur_mathematical_equivalence(self):
        """Test that separable 1D Blur is mathematically equivalent to 2D Blur within a tiny tolerance."""
        image = self.create_test_image(batch_size=1, height=32, width=32, channels=3)
        channels = 3
        blur_radius = 5
        sigma = 2.0
        kernel_size = blur_radius * 2 + 1

        # Compute manually using original 2D method
        kernel_2d = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)
        tensor_image = image.permute(0, 3, 1, 2)
        padded_image = F.pad(tensor_image, (blur_radius, blur_radius, blur_radius, blur_radius), 'reflect')
        expected = F.conv2d(padded_image, kernel_2d, groups=channels)
        expected = expected.permute(0, 2, 3, 1)

        # Run optimized Blur node
        blur_node = Blur()
        actual = blur_node.execute(image, blur_radius=blur_radius, sigma=sigma)[0]

        diff = torch.abs(expected - actual).max()
        assert diff.item() < 1e-6

    def test_sharpen_zero_or_negative_radius(self):
        """Test that Sharpen returns the input unchanged when radius is 0 or negative."""
        image = self.create_test_image()
        sharpen_node = Sharpen()

        # Zero radius
        out_zero = sharpen_node.execute(image, sharpen_radius=0, sigma=1.0, alpha=1.0)
        assert torch.equal(out_zero[0], image)

        # Negative radius
        out_neg = sharpen_node.execute(image, sharpen_radius=-2, sigma=1.0, alpha=1.0)
        assert torch.equal(out_neg[0], image)

    def test_sharpen_shape_preservation(self):
        """Test that Sharpen preserves dimensions and clamps output between 0.0 and 1.0."""
        image = self.create_test_image(batch_size=1, height=32, width=32, channels=3)
        sharpen_node = Sharpen()

        out = sharpen_node.execute(image, sharpen_radius=3, sigma=1.0, alpha=1.5)[0]
        assert out.shape == image.shape
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_latent_sharpen_shape_preservation(self):
        """Test that LatentOperationSharpen node works and preserves shape."""
        latent = torch.rand(1, 4, 16, 16) # typical latent shape
        latent_node = LatentOperationSharpen()

        # Calling execute returns a dict/tuple with a sharpen function
        sharpen_func = latent_node.execute(sharpen_radius=2, sigma=1.0, alpha=0.1)[0]
        result = sharpen_func(latent)

        assert result.shape == latent.shape
