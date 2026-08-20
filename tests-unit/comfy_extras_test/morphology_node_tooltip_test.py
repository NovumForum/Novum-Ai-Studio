import unittest
from unittest.mock import patch, MagicMock
import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_cli_args = MagicMock()
mock_cli_args.args.cpu = True
mock_cli_args.args.deterministic = False
mock_cli_args.args.directml = None
mock_cli_args.args.lowvram = False
mock_cli_args.args.novram = False
mock_cli_args.args.highvram = False
mock_cli_args.args.gpu_only = False
mock_cli_args.args.force_fp32 = False
mock_cli_args.args.disable_xformers = True
mock_cli_args.args.use_pytorch_cross_attention = False
mock_cli_args.args.use_split_cross_attention = False
mock_cli_args.args.use_quad_cross_attention = False
mock_cli_args.args.supports_fp8_compute = False
mock_cli_args.args.fast = []
mock_cli_args.args.disable_smart_memory = False
mock_cli_args.args.reserve_vram = None
mock_cli_args.args.async_offload = None
mock_cli_args.args.disable_async_offload = True
mock_cli_args.args.disable_pinned_memory = True

with patch.dict('sys.modules', {'nodes': mock_nodes, 'comfy.cli_args': mock_cli_args}):
    from comfy_extras.nodes_morphology import Morphology, ImageRGBToYUV, ImageYUVToRGB

class TestMorphologyNodeUX(unittest.TestCase):
    def test_morphology_schema(self):
        schema = Morphology.define_schema()
        self.assertEqual(schema.node_id, "Morphology")
        self.assertEqual(schema.display_name, "ImageMorphology")
        self.assertTrue(len(schema.description) > 0)
        self.assertIn("erode", schema.search_aliases)
        self.assertIn("dilate", schema.search_aliases)
        self.assertIn("morphology", schema.search_aliases)

        # Inputs
        inputs = {inp.id: inp for inp in schema.inputs}
        self.assertIn("image", inputs)
        self.assertIn("operation", inputs)
        self.assertIn("kernel_size", inputs)

        self.assertTrue(len(inputs["image"].tooltip) > 0)
        self.assertTrue(len(inputs["operation"].tooltip) > 0)
        self.assertTrue(len(inputs["kernel_size"].tooltip) > 0)

        # Outputs
        outputs = {out.id: out for out in schema.outputs}
        self.assertIn("IMAGE", outputs)
        self.assertTrue(len(outputs["IMAGE"].tooltip) > 0)

    def test_rgb_to_yuv_schema(self):
        schema = ImageRGBToYUV.define_schema()
        self.assertEqual(schema.node_id, "ImageRGBToYUV")
        self.assertEqual(schema.display_name, "RGB to YUV")
        self.assertTrue(len(schema.description) > 0)
        self.assertIn("color space conversion", schema.search_aliases)
        self.assertIn("yuv split", schema.search_aliases)

        # Inputs
        inputs = {inp.id: inp for inp in schema.inputs}
        self.assertIn("image", inputs)
        self.assertTrue(len(inputs["image"].tooltip) > 0)

        # Outputs
        outputs = {out.id: out for out in schema.outputs}
        self.assertIn("Y", outputs)
        self.assertIn("U", outputs)
        self.assertIn("V", outputs)
        self.assertTrue(len(outputs["Y"].tooltip) > 0)
        self.assertTrue(len(outputs["U"].tooltip) > 0)
        self.assertTrue(len(outputs["V"].tooltip) > 0)

    def test_yuv_to_rgb_schema(self):
        schema = ImageYUVToRGB.define_schema()
        self.assertEqual(schema.node_id, "ImageYUVToRGB")
        self.assertEqual(schema.display_name, "YUV to RGB")
        self.assertTrue(len(schema.description) > 0)
        self.assertIn("color space conversion", schema.search_aliases)
        self.assertIn("yuv merge", schema.search_aliases)

        # Inputs
        inputs = {inp.id: inp for inp in schema.inputs}
        self.assertIn("Y", inputs)
        self.assertIn("U", inputs)
        self.assertIn("V", inputs)
        self.assertTrue(len(inputs["Y"].tooltip) > 0)
        self.assertTrue(len(inputs["U"].tooltip) > 0)
        self.assertTrue(len(inputs["V"].tooltip) > 0)

        # Outputs
        outputs = {out.id: out for out in schema.outputs}
        self.assertIn("IMAGE", outputs)
        self.assertTrue(len(outputs["IMAGE"].tooltip) > 0)

    def test_morphology_execution(self):
        img = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
        img[0, 6:10, 6:10, :] = 1.0
        res = Morphology.execute(img, "dilate", 3)
        self.assertEqual(res[0].shape, (1, 16, 16, 3))

    def test_yuv_roundtrip_execution(self):
        img = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        yuv_res = ImageRGBToYUV.execute(img)
        self.assertEqual(yuv_res[0].shape, (1, 8, 8, 3))
        self.assertEqual(yuv_res[1].shape, (1, 8, 8, 3))
        self.assertEqual(yuv_res[2].shape, (1, 8, 8, 3))
        rgb_res = ImageYUVToRGB.execute(yuv_res[0], yuv_res[1], yuv_res[2])
        self.assertEqual(rgb_res[0].shape, (1, 8, 8, 3))


if __name__ == "__main__":
    unittest.main()
