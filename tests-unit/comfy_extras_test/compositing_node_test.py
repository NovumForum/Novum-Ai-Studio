import unittest
import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


class TestCompositingNodes(unittest.TestCase):
    def test_split_image_with_alpha_rgba(self):
        batch_size, h, w = 4, 32, 32
        # Create dummy RGBA tensor [B, H, W, 4]
        rgba_image = torch.rand(batch_size, h, w, 4)
        out_image, out_mask = SplitImageWithAlpha.execute(rgba_image)

        # Expected shape & outputs
        self.assertEqual(out_image.shape, (batch_size, h, w, 3))
        self.assertEqual(out_mask.shape, (batch_size, h, w))
        self.assertTrue(torch.allclose(out_image, rgba_image[:, :, :, :3]))
        self.assertTrue(torch.allclose(out_mask, 1.0 - rgba_image[:, :, :, 3]))

    def test_split_image_with_alpha_rgb(self):
        batch_size, h, w = 4, 32, 32
        # Create dummy RGB tensor [B, H, W, 3]
        rgb_image = torch.rand(batch_size, h, w, 3)
        out_image, out_mask = SplitImageWithAlpha.execute(rgb_image)

        # Expected shape & outputs
        self.assertEqual(out_image.shape, (batch_size, h, w, 3))
        self.assertEqual(out_mask.shape, (batch_size, h, w))
        self.assertTrue(torch.allclose(out_image, rgb_image))
        self.assertTrue(torch.allclose(out_mask, torch.zeros(batch_size, h, w)))

    def test_join_image_with_alpha_matching_dims(self):
        batch_size, h, w = 4, 32, 32
        rgb_image = torch.rand(batch_size, h, w, 3)
        mask = torch.rand(batch_size, h, w)

        out_image = JoinImageWithAlpha.execute(rgb_image, mask)[0]

        self.assertEqual(out_image.shape, (batch_size, h, w, 4))
        self.assertTrue(torch.allclose(out_image[:, :, :, :3], rgb_image))
        self.assertTrue(torch.allclose(out_image[:, :, :, 3], 1.0 - mask))

    def test_join_image_with_alpha_mismatched_batch_size(self):
        image_batch_size, mask_batch_size = 4, 2
        h, w = 32, 32
        rgb_image = torch.rand(image_batch_size, h, w, 3)
        mask = torch.rand(mask_batch_size, h, w)

        out_image = JoinImageWithAlpha.execute(rgb_image, mask)[0]

        expected_batch_size = min(image_batch_size, mask_batch_size)
        self.assertEqual(out_image.shape, (expected_batch_size, h, w, 4))
        self.assertTrue(torch.allclose(out_image[:, :, :, :3], rgb_image[:expected_batch_size]))
        self.assertTrue(torch.allclose(out_image[:, :, :, 3], 1.0 - mask[:expected_batch_size]))

    def test_join_image_with_alpha_mismatched_spatial_dims(self):
        batch_size = 2
        rgb_image = torch.rand(batch_size, 64, 64, 3)
        mask = torch.rand(batch_size, 32, 32)

        out_image = JoinImageWithAlpha.execute(rgb_image, mask)[0]

        self.assertEqual(out_image.shape, (batch_size, 64, 64, 4))
        self.assertTrue(torch.allclose(out_image[:, :, :, :3], rgb_image))


if __name__ == "__main__":
    unittest.main()
