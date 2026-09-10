import pytest
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_post_processing import Blend, Blur, Quantize, Sharpen, ImageScaleToTotalPixels

def test_blend_schema_metadata_and_tooltips():
    schema = Blend.define_schema()
    assert schema.node_id == "ImageBlend"
    assert schema.display_name == "Image Blend"
    assert schema.category == "image/postprocessing"
    assert "Blends two images together" in schema.description
    assert "blend" in schema.search_aliases
    assert "composite" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert "image1" in input_map
    assert "image2" in input_map
    assert "blend_factor" in input_map
    assert "blend_mode" in input_map

    assert input_map["image1"].tooltip == "Base background image tensor."
    assert input_map["image2"].tooltip == "Foreground image tensor to blend over image1."
    assert "Opacity / weight factor" in input_map["blend_factor"].tooltip
    assert "Blend mode algorithm" in input_map["blend_mode"].tooltip

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "The resulting blended image tensor."

def test_blur_schema_metadata_and_tooltips():
    schema = Blur.define_schema()
    assert schema.node_id == "ImageBlur"
    assert schema.display_name == "Image Blur"
    assert schema.category == "image/postprocessing"
    assert "Applies Gaussian blur convolution" in schema.description
    assert "gaussian blur" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert input_map["image"].tooltip == "The input image tensor to blur."
    assert "Radius of the Gaussian blur kernel" in input_map["blur_radius"].tooltip
    assert "Standard deviation of the Gaussian kernel" in input_map["sigma"].tooltip

    assert schema.outputs[0].tooltip == "The blurred image tensor."

def test_quantize_schema_metadata_and_tooltips():
    schema = Quantize.define_schema()
    assert schema.node_id == "ImageQuantize"
    assert schema.display_name == "Image Quantize"
    assert schema.category == "image/postprocessing"
    assert "Reduces the total number of colors" in schema.description
    assert "dither" in schema.search_aliases
    assert "floyd-steinberg" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert input_map["image"].tooltip == "The input image tensor to quantize."
    assert "Target maximum number of palette colors" in input_map["colors"].tooltip
    assert "Dithering method" in input_map["dither"].tooltip

    assert schema.outputs[0].tooltip == "The color-quantized image tensor."

def test_sharpen_schema_metadata_and_tooltips():
    schema = Sharpen.define_schema()
    assert schema.node_id == "ImageSharpen"
    assert schema.display_name == "Image Sharpen"
    assert schema.category == "image/postprocessing"
    assert "Sharpens an image using an unsharp mask filter" in schema.description
    assert "unsharp mask" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert input_map["image"].tooltip == "The input image tensor to sharpen."
    assert "Radius of the sharpening kernel" in input_map["sharpen_radius"].tooltip
    assert "Standard deviation of the Gaussian kernel" in input_map["sigma"].tooltip
    assert "Sharpening intensity factor" in input_map["alpha"].tooltip

    assert schema.outputs[0].tooltip == "The sharpened image tensor."

def test_scale_to_total_pixels_schema_metadata_and_tooltips():
    schema = ImageScaleToTotalPixels.define_schema()
    assert schema.node_id == "ImageScaleToTotalPixels"
    assert schema.display_name == "Scale Image to Megapixels"
    assert schema.category == "image/upscaling"
    assert "total pixel count matches a target megapixel budget" in schema.description
    assert "pixel budget" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert input_map["image"].tooltip == "The input image tensor to scale."
    assert "Interpolation algorithm" in input_map["upscale_method"].tooltip
    assert "Target pixel budget in megapixels" in input_map["megapixels"].tooltip
    assert "Rounding step size in pixels" in input_map["resolution_steps"].tooltip

    assert schema.outputs[0].tooltip == "The resized image tensor scaled to the target megapixel budget."
