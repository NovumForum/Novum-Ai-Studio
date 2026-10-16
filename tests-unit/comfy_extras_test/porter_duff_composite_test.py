import torch
from unittest.mock import patch, MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_compositing import PorterDuffImageComposite, PorterDuffMode, porter_duff_composite


class TestPorterDuffImageComposite:

    def create_test_image(self, batch_size=2, height=64, width=64, channels=3):
        return torch.rand(batch_size, height, width, channels)

    def create_test_mask(self, batch_size=2, height=64, width=64):
        return torch.rand(batch_size, height, width)

    def test_basic_composite_matching_dimensions(self):
        source = self.create_test_image(batch_size=2, height=32, width=32)
        source_alpha = self.create_test_mask(batch_size=2, height=32, width=32)
        destination = self.create_test_image(batch_size=2, height=32, width=32)
        destination_alpha = self.create_test_mask(batch_size=2, height=32, width=32)

        result = PorterDuffImageComposite.execute(source, source_alpha, destination, destination_alpha, "SRC_OVER")

        out_image = result[0]
        out_mask = result[1]

        assert out_image.shape == (2, 32, 32, 3)
        assert out_mask.shape == (2, 32, 32)
        assert torch.all(out_image >= 0.0) and torch.all(out_image <= 1.0)
        assert torch.all(out_mask >= 0.0) and torch.all(out_mask <= 1.0)

    def test_composite_mismatched_batch_sizes(self):
        source = self.create_test_image(batch_size=4, height=32, width=32)
        source_alpha = self.create_test_mask(batch_size=3, height=32, width=32)
        destination = self.create_test_image(batch_size=2, height=32, width=32)
        destination_alpha = self.create_test_mask(batch_size=2, height=32, width=32)

        result = PorterDuffImageComposite.execute(source, source_alpha, destination, destination_alpha, "DST_OVER")

        # Result batch size should be constrained by min(len(inputs)) = 2
        out_image = result[0]
        out_mask = result[1]

        assert out_image.shape == (2, 32, 32, 3)
        assert out_mask.shape == (2, 32, 32)

    def test_composite_mismatched_spatial_dimensions(self):
        source = self.create_test_image(batch_size=2, height=16, width=16)
        source_alpha = self.create_test_mask(batch_size=2, height=16, width=16)
        destination = self.create_test_image(batch_size=2, height=32, width=32)
        destination_alpha = self.create_test_mask(batch_size=2, height=24, width=24)

        result = PorterDuffImageComposite.execute(source, source_alpha, destination, destination_alpha, "ADD")

        out_image = result[0]
        out_mask = result[1]

        # Resized spatial dimensions should all match destination (32x32)
        assert out_image.shape == (2, 32, 32, 3)
        assert out_mask.shape == (2, 32, 32)

    def test_numerical_equivalence_against_unvectorized_reference(self):
        torch.manual_seed(42)
        batch_size = 3
        source = self.create_test_image(batch_size=batch_size, height=32, width=32)
        source_alpha = self.create_test_mask(batch_size=batch_size, height=32, width=32)
        destination = self.create_test_image(batch_size=batch_size, height=32, width=32)
        destination_alpha = self.create_test_mask(batch_size=batch_size, height=32, width=32)

        # Unvectorized reference computation
        ref_out_images = []
        ref_out_alphas = []
        for i in range(batch_size):
            src_i = source[i]
            dst_i = destination[i]
            src_a = source_alpha[i].unsqueeze(2)
            dst_a = destination_alpha[i].unsqueeze(2)
            res_img, res_a = porter_duff_composite(src_i, src_a, dst_i, dst_a, PorterDuffMode["SRC_OVER"])
            ref_out_images.append(res_img)
            ref_out_alphas.append(res_a.squeeze(2))

        ref_image = torch.stack(ref_out_images)
        ref_mask = torch.stack(ref_out_alphas)

        vec_result = PorterDuffImageComposite.execute(source, source_alpha, destination, destination_alpha, "SRC_OVER")

        torch.testing.assert_close(vec_result[0], ref_image)
        torch.testing.assert_close(vec_result[1], ref_mask)
