import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock dependencies
mock_torch = MagicMock()
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_folder_paths = MagicMock()
mock_folder_paths.get_filename_list.return_value = ["test_upscale_model.safetensors"]

mock_node_helpers = MagicMock()
mock_comfy = MagicMock()
mock_packaging = MagicMock()
mock_numpy = MagicMock()
mock_pil = MagicMock()
mock_tqdm = MagicMock()

modules_to_patch = {
    'torch': mock_torch,
    'nodes': mock_nodes,
    'node_helpers': mock_node_helpers,
    'folder_paths': mock_folder_paths,
    'comfy': mock_comfy,
    'comfy.cli_args': MagicMock(),
    'comfy.model_management': MagicMock(),
    'comfy.ldm': MagicMock(),
    'comfy.ldm.hunyuan_video': MagicMock(),
    'comfy.ldm.hunyuan_video.upsampler': MagicMock(),
    'comfy.ldm.lightricks': MagicMock(),
    'comfy.ldm.lightricks.latent_upsampler': MagicMock(),
    'packaging': mock_packaging,
    'packaging.version': mock_packaging,
    'av': MagicMock(),
    'av.container': MagicMock(),
    'av.subtitles': MagicMock(),
    'av.subtitles.stream': MagicMock(),
    'numpy': mock_numpy,
    'PIL': mock_pil,
    'PIL.Image': mock_pil,
    'PIL.PngImagePlugin': mock_pil,
    'tqdm': mock_tqdm,
}

with patch.dict('sys.modules', modules_to_patch):
    from comfy_extras.nodes_hunyuan import (
        CLIPTextEncodeHunyuanDiT,
        EmptyHunyuanLatentVideo,
        EmptyHunyuanVideo15Latent,
        HunyuanVideo15ImageToVideo,
        HunyuanVideo15SuperResolution,
        LatentUpscaleModelLoader,
        HunyuanVideo15LatentUpscaleWithModel,
        TextEncodeHunyuanVideo_ImageToVideo,
        HunyuanImageToVideo,
        EmptyHunyuanImageLatent,
        HunyuanRefinerLatent,
    )


class TestHunyuanNodeTooltips(unittest.TestCase):

    def test_clip_text_encode_hunyuan_dit_schema(self):
        schema = CLIPTextEncodeHunyuanDiT.define_schema()
        self.assertEqual(schema.node_id, "CLIPTextEncodeHunyuanDiT")
        self.assertEqual(schema.display_name, "CLIP Text Encode (HunyuanDiT)")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

        # Inputs
        input_dict = {inp.id: inp for inp in schema.inputs}
        self.assertIn("clip", input_dict)
        self.assertIn("bert", input_dict)
        self.assertIn("mt5xl", input_dict)
        self.assertTrue(hasattr(input_dict["clip"], "tooltip"))

        # Outputs
        self.assertEqual(len(schema.outputs), 1)
        self.assertTrue(hasattr(schema.outputs[0], "tooltip"))

    def test_empty_hunyuan_latent_video_schema(self):
        schema = EmptyHunyuanLatentVideo.define_schema()
        self.assertEqual(schema.node_id, "EmptyHunyuanLatentVideo")
        self.assertEqual(schema.display_name, "Empty HunyuanVideo 1.0 Latent")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

        input_dict = {inp.id: inp for inp in schema.inputs}
        for field in ["width", "height", "length", "batch_size"]:
            self.assertIn(field, input_dict)
            self.assertTrue(hasattr(input_dict[field], "tooltip"))

    def test_empty_hunyuan_video15_latent_schema(self):
        schema = EmptyHunyuanVideo15Latent.define_schema()
        self.assertEqual(schema.node_id, "EmptyHunyuanVideo15Latent")
        self.assertEqual(schema.display_name, "Empty HunyuanVideo 1.5 Latent")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_hunyuan_video15_image_to_video_schema(self):
        schema = HunyuanVideo15ImageToVideo.define_schema()
        self.assertEqual(schema.node_id, "HunyuanVideo15ImageToVideo")
        self.assertEqual(schema.display_name, "HunyuanVideo 1.5 Image to Video")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

        input_dict = {inp.id: inp for inp in schema.inputs}
        for field in ["positive", "negative", "vae", "width", "height", "length", "batch_size", "start_image", "clip_vision_output"]:
            self.assertIn(field, input_dict)
            self.assertTrue(hasattr(input_dict[field], "tooltip"))

    def test_hunyuan_video15_super_resolution_schema(self):
        schema = HunyuanVideo15SuperResolution.define_schema()
        self.assertEqual(schema.node_id, "HunyuanVideo15SuperResolution")
        self.assertEqual(schema.display_name, "HunyuanVideo 1.5 Super Resolution")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_latent_upscale_model_loader_schema(self):
        schema = LatentUpscaleModelLoader.define_schema()
        self.assertEqual(schema.node_id, "LatentUpscaleModelLoader")
        self.assertEqual(schema.display_name, "Load Latent Upscale Model")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_hunyuan_video15_latent_upscale_with_model_schema(self):
        schema = HunyuanVideo15LatentUpscaleWithModel.define_schema()
        self.assertEqual(schema.node_id, "HunyuanVideo15LatentUpscaleWithModel")
        self.assertEqual(schema.display_name, "Hunyuan Video 15 Latent Upscale With Model")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_text_encode_hunyuan_video_i2v_schema(self):
        schema = TextEncodeHunyuanVideo_ImageToVideo.define_schema()
        self.assertEqual(schema.node_id, "TextEncodeHunyuanVideo_ImageToVideo")
        self.assertEqual(schema.display_name, "Text Encode HunyuanVideo Image to Video")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_hunyuan_image_to_video_schema(self):
        schema = HunyuanImageToVideo.define_schema()
        self.assertEqual(schema.node_id, "HunyuanImageToVideo")
        self.assertEqual(schema.display_name, "Hunyuan Image to Video")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_empty_hunyuan_image_latent_schema(self):
        schema = EmptyHunyuanImageLatent.define_schema()
        self.assertEqual(schema.node_id, "EmptyHunyuanImageLatent")
        self.assertEqual(schema.display_name, "Empty Hunyuan Image Latent")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)

    def test_hunyuan_refiner_latent_schema(self):
        schema = HunyuanRefinerLatent.define_schema()
        self.assertEqual(schema.node_id, "HunyuanRefinerLatent")
        self.assertEqual(schema.display_name, "Hunyuan Refiner Latent")
        self.assertTrue(hasattr(schema, "description") and len(schema.description) > 0)
        self.assertTrue(hasattr(schema, "search_aliases") and len(schema.search_aliases) > 0)


if __name__ == "__main__":
    unittest.main()
