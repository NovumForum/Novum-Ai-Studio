import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import comfy.cli_args
comfy.cli_args.args.cpu = True

from nodes import ImagePadForOutpaint

class TestImagePadForOutpaint(unittest.TestCase):
    def setUp(self):
        self.node = ImagePadForOutpaint()

    def test_expand_image_basic(self):
        image = torch.ones((1, 100, 100, 3), dtype=torch.float32)
        padded_img, mask = self.node.expand_image(image, left=20, top=10, right=30, bottom=40, feathering=10)

        self.assertEqual(padded_img.shape, (1, 150, 150, 3))
        self.assertEqual(mask.shape, (1, 150, 150))
        # Center image pixels preserved
        self.assertTrue(torch.allclose(padded_img[:, 10:110, 20:120, :], image))

    def test_expand_image_no_feathering(self):
        image = torch.ones((1, 64, 64, 3), dtype=torch.float32)
        padded_img, mask = self.node.expand_image(image, left=10, top=10, right=10, bottom=10, feathering=0)

        self.assertEqual(padded_img.shape, (1, 84, 84, 3))
        self.assertEqual(mask.shape, (1, 84, 84))
        # Inner mask block should be zeros when feathering is 0
        self.assertTrue(torch.allclose(mask[:, 10:74, 10:74], torch.zeros((1, 64, 64))))

    def test_expand_image_zero_padding(self):
        image = torch.ones((1, 64, 64, 3), dtype=torch.float32)
        padded_img, mask = self.node.expand_image(image, left=0, top=0, right=0, bottom=0, feathering=10)

        self.assertEqual(padded_img.shape, (1, 64, 64, 3))
        self.assertEqual(mask.shape, (1, 64, 64))

    def test_expand_image_partial_padding(self):
        image = torch.rand((1, 128, 128, 3), dtype=torch.float32)
        # Top and left padded, right and bottom zero
        padded_img, mask = self.node.expand_image(image, left=16, top=16, right=0, bottom=0, feathering=20)

        self.assertEqual(padded_img.shape, (1, 144, 144, 3))
        self.assertEqual(mask.shape, (1, 144, 144))

if __name__ == "__main__":
    unittest.main()
