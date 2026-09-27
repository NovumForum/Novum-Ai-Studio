import unittest
import torch


class TestImageYUVToRGB(unittest.TestCase):
    def test_image_yuv_to_rgb_single_channel(self):
        from comfy_extras.nodes_morphology import ImageRGBToYUV, ImageYUVToRGB

        # Create mock RGB image [1, 64, 64, 3]
        rgb_input = torch.rand((1, 64, 64, 3), dtype=torch.float32)

        # Convert to YUV
        rgb_to_yuv = ImageRGBToYUV.execute(rgb_input)
        y, u, v = rgb_to_yuv.args

        # Convert back to RGB using ImageYUVToRGB
        yuv_to_rgb = ImageYUVToRGB.execute(y, u, v)
        rgb_output = yuv_to_rgb.args[0]

        # Check shape
        self.assertEqual(rgb_output.shape, (1, 64, 64, 3))

        # Compare against unoptimized baseline (mean applied explicitly)
        y_chan_mean = torch.mean(y, dim=-1, keepdim=True)
        u_chan_mean = torch.mean(u, dim=-1, keepdim=True)
        v_chan_mean = torch.mean(v, dim=-1, keepdim=True)
        image_mean = torch.cat([y_chan_mean, u_chan_mean, v_chan_mean], dim=-1)
        import kornia.color

        expected_out = kornia.color.ycbcr_to_rgb(image_mean.movedim(-1, 1)).movedim(1, -1)

        torch.testing.assert_close(rgb_output, expected_out)

    def test_image_yuv_to_rgb_multichannel(self):
        from comfy_extras.nodes_morphology import ImageYUVToRGB

        # Create 3-channel input tensors for Y, U, V
        y = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        u = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        v = torch.rand((1, 32, 32, 3), dtype=torch.float32)

        yuv_to_rgb = ImageYUVToRGB.execute(y, u, v)
        rgb_output = yuv_to_rgb.args[0]

        self.assertEqual(rgb_output.shape, (1, 32, 32, 3))


if __name__ == "__main__":
    unittest.main()
