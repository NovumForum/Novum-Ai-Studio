import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True

from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_compositing import SplitImageWithAlpha


class TestSplitImageWithAlpha:
    def test_split_image_4channel_rgba(self):
        """Test splitting a 4-channel (RGBA) image batch tensor into RGB and mask outputs."""
        batch_size, h, w = 4, 32, 32
        image_rgba = torch.rand(batch_size, h, w, 4)

        result = SplitImageWithAlpha.execute(image_rgba)

        out_image, out_mask = result.result
        assert out_image.shape == (batch_size, h, w, 3)
        assert out_mask.shape == (batch_size, h, w)

        # Verify RGB channels match original image first 3 channels
        assert torch.equal(out_image, image_rgba[..., :3])
        # Verify mask is 1.0 - alpha channel (4th channel)
        expected_mask = 1.0 - image_rgba[..., 3]
        assert torch.equal(out_mask, expected_mask)

    def test_split_image_3channel_rgb(self):
        """Test splitting a 3-channel (RGB) image tensor creates default full opacity mask."""
        batch_size, h, w = 2, 16, 16
        image_rgb = torch.rand(batch_size, h, w, 3)

        result = SplitImageWithAlpha.execute(image_rgb)

        out_image, out_mask = result.result
        assert out_image.shape == (batch_size, h, w, 3)
        assert out_mask.shape == (batch_size, h, w)

        assert torch.equal(out_image, image_rgb)
        # For 3-channel image, mask is 1.0 - 1.0 = 0.0 (full opacity mask)
        assert torch.equal(out_mask, torch.zeros(batch_size, h, w))

    def test_split_image_single_batch(self):
        """Test splitting a single-frame batch (batch_size=1) 4-channel image tensor."""
        image_rgba = torch.rand(1, 64, 64, 4)

        result = SplitImageWithAlpha.execute(image_rgba)

        out_image, out_mask = result.result
        assert out_image.shape == (1, 64, 64, 3)
        assert out_mask.shape == (1, 64, 64)

    def test_split_image_numerical_equivalence_with_reference(self):
        """Test that vectorized output is exactly equal to unvectorized reference unrolling."""
        image_rgba = torch.rand(8, 48, 48, 4)

        # Reference unvectorized implementation
        ref_images = [i[:, :, :3] for i in image_rgba]
        ref_alphas = [i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image_rgba]
        expected_images = torch.stack(ref_images)
        expected_mask = 1.0 - torch.stack(ref_alphas)

        result = SplitImageWithAlpha.execute(image_rgba)
        out_image, out_mask = result.result

        assert torch.equal(out_image, expected_images)
        assert torch.equal(out_mask, expected_mask)
