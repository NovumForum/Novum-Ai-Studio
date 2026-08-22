import torch

import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_post_processing import Blend, Blur, Quantize, Sharpen, ImageScaleToTotalPixels


class TestPostProcessingNodeTooltipsAndSchema:

    def test_blend_schema(self):
        schema = Blend.define_schema()
        assert schema.node_id == "ImageBlend"
        assert schema.display_name == "Image Blend"
        assert "Blend two images together" in schema.description
        assert "mix images" in schema.search_aliases
        assert "composite images" in schema.search_aliases

        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert inputs_dict["image1"].tooltip == "The base background image."
        assert inputs_dict["image2"].tooltip == "The foreground image to blend over image1."
        assert "Blending opacity weight" in inputs_dict["blend_factor"].tooltip
        assert "Blending mode function" in inputs_dict["blend_mode"].tooltip

        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The blended output image."

    def test_blur_schema(self):
        schema = Blur.define_schema()
        assert schema.node_id == "ImageBlur"
        assert schema.display_name == "Image Blur"
        assert "Apply Gaussian blur filter" in schema.description
        assert "gaussian blur" in schema.search_aliases
        assert "defocus" in schema.search_aliases

        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert inputs_dict["image"].tooltip == "The input image to blur."
        assert "Radius of the blur kernel" in inputs_dict["blur_radius"].tooltip
        assert "Standard deviation of the Gaussian kernel" in inputs_dict["sigma"].tooltip

        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The blurred output image."

    def test_quantize_schema(self):
        schema = Quantize.define_schema()
        assert schema.node_id == "ImageQuantize"
        assert schema.display_name == "Image Quantize"
        assert "Reduce the color palette size" in schema.description
        assert "dither" in schema.search_aliases
        assert "color reduction" in schema.search_aliases

        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert inputs_dict["image"].tooltip == "The input image to quantize."
        assert "Maximum number of distinct colors" in inputs_dict["colors"].tooltip
        assert "Dithering method to reduce color banding" in inputs_dict["dither"].tooltip

        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The color-quantized output image."

    def test_sharpen_schema(self):
        schema = Sharpen.define_schema()
        assert schema.node_id == "ImageSharpen"
        assert schema.display_name == "Image Sharpen"
        assert "Enhance detail and image sharpness" in schema.description
        assert "unsharp mask" in schema.search_aliases
        assert "clarity" in schema.search_aliases

        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert inputs_dict["image"].tooltip == "The input image to sharpen."
        assert "Radius of the sharpening filter kernel" in inputs_dict["sharpen_radius"].tooltip
        assert "Standard deviation of the Gaussian filter" in inputs_dict["sigma"].tooltip
        assert "Strength factor of the sharpening filter" in inputs_dict["alpha"].tooltip

        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The sharpened output image."

    def test_image_scale_to_total_pixels_schema(self):
        schema = ImageScaleToTotalPixels.define_schema()
        assert schema.node_id == "ImageScaleToTotalPixels"
        assert schema.display_name == "Scale Image to Total Pixels"
        assert "Scale an image so its total pixel count matches" in schema.description
        assert "scale megapixels" in schema.search_aliases
        assert "resize megapixels" in schema.search_aliases

        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert inputs_dict["image"].tooltip == "The input image to resize."
        assert "Interpolation algorithm used for scaling" in inputs_dict["upscale_method"].tooltip
        assert "Target total resolution in megapixels" in inputs_dict["megapixels"].tooltip
        assert "Enforce scaled width and height" in inputs_dict["resolution_steps"].tooltip

        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The resized output image."

    def test_blend_execution(self):
        img1 = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.2
        img2 = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.8
        out = Blend.execute(img1, img2, blend_factor=0.5, blend_mode="normal")
        assert out[0].shape == (1, 32, 32, 3)
        assert torch.allclose(out[0], torch.tensor(0.5), atol=1e-3)

    def test_blur_execution(self):
        img = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        out = Blur.execute(img, blur_radius=2, sigma=1.0)
        assert out[0].shape == (1, 32, 32, 3)

    def test_quantize_execution(self):
        img = torch.rand((1, 16, 16, 3), dtype=torch.float32)
        out = Quantize.execute(img, colors=8, dither="none")
        assert out[0].shape == (1, 16, 16, 3)

    def test_sharpen_execution(self):
        img = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        out = Sharpen.execute(img, sharpen_radius=1, sigma=1.0, alpha=1.0)
        assert out[0].shape == (1, 32, 32, 3)

    def test_scale_to_total_pixels_execution(self):
        img = torch.rand((1, 64, 64, 3), dtype=torch.float32)
        out = ImageScaleToTotalPixels.execute(img, upscale_method="bilinear", megapixels=0.01, resolution_steps=1)
        assert out[0].shape[0] == 1
        assert out[0].shape[3] == 3
