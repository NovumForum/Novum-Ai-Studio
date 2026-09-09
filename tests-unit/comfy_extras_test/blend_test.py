import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_post_processing import Blend


class TestBlend:
    def create_image(self, batch_size=2, height=32, width=32, channels=3, value=None):
        if value is not None:
            return torch.full((batch_size, height, width, channels), value, dtype=torch.float32)
        return torch.rand((batch_size, height, width, channels), dtype=torch.float32)

    def test_zero_blend_factor_fast_path(self):
        """Test that blend_factor=0.0 returns image1 directly."""
        image1 = self.create_image(value=0.2)
        image2 = self.create_image(value=0.8)

        output = Blend.execute(image1, image2, blend_factor=0.0, blend_mode="multiply")
        result = output.result[0]

        assert torch.equal(result, image1)

    def test_full_blend_factor_normal(self):
        """Test that blend_factor=1.0 with mode 'normal' returns image2."""
        image1 = self.create_image(value=0.2)
        image2 = self.create_image(value=0.8)

        output = Blend.execute(image1, image2, blend_factor=1.0, blend_mode="normal")
        result = output.result[0]

        assert torch.allclose(result, image2, atol=1e-5)

    def test_blend_modes(self):
        """Test all blend modes with blend_factor=0.5."""
        image1 = self.create_image(value=0.4)
        image2 = self.create_image(value=0.6)
        modes = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"]

        for mode in modes:
            output = Blend.execute(image1, image2, blend_factor=0.5, blend_mode=mode)
            result = output.result[0]

            assert result.shape == image1.shape
            assert torch.all(result >= 0.0) and torch.all(result <= 1.0)

    def test_multiply_blend_mode_numerical_correctness(self):
        """Test numerical correctness for multiply blend mode."""
        image1 = self.create_image(value=0.4)
        image2 = self.create_image(value=0.6)
        # multiply = img1 * img2 = 0.24
        # lerp(0.4, 0.24, 0.5) = 0.32
        expected_val = 0.32

        output = Blend.execute(image1, image2, blend_factor=0.5, blend_mode="multiply")
        result = output.result[0]

        assert torch.allclose(result, torch.tensor(expected_val), atol=1e-5)

    def test_shape_mismatch_upscaling(self):
        """Test that mismatched image shapes are handled correctly."""
        image1 = self.create_image(height=64, width=64)
        image2 = self.create_image(height=32, width=32)

        output = Blend.execute(image1, image2, blend_factor=0.5, blend_mode="normal")
        result = output.result[0]

        assert result.shape == (2, 64, 64, 3)

    def test_alpha_channel_blending(self):
        """Test blending images with alpha channel (4 channels)."""
        image1 = self.create_image(channels=4, value=0.5)
        image2 = self.create_image(channels=4, value=0.8)

        output = Blend.execute(image1, image2, blend_factor=0.5, blend_mode="multiply")
        result = output.result[0]

        assert result.shape[-1] == 4
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0)
