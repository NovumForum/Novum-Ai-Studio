from unittest.mock import MagicMock
import torch

# Mock CUDA functions and memory stats for CPU-only sandbox environment
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda dev: MagicMock(total_memory=8 * 1024 * 1024 * 1024)
torch.cuda.memory_stats = lambda dev=None: {"reserved_bytes.all.current": 0}
torch.cuda.mem_get_info = lambda dev=None: (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)

from comfy_extras.nodes_mask import (
    LatentCompositeMasked,
    ImageCompositeMasked,
    MaskToImage,
    ImageToMask,
    ImageColorToMask,
    SolidMask,
    InvertMask,
    CropMask,
    MaskComposite,
    FeatherMask,
    GrowMask,
    ThresholdMask,
    MaskPreview,
)


def test_latent_composite_masked_schema():
    schema = LatentCompositeMasked.define_schema()
    assert schema.node_id == "LatentCompositeMasked"
    assert schema.display_name == "Latent Composite Masked"
    assert schema.description == "Composites a source latent into a destination latent using an optional mask and offset coordinates."
    assert "composite latent" in schema.search_aliases
    assert "overlay latent" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["destination"].tooltip == "The background latent image into which the source latent will be composited."
    assert inputs["source"].tooltip == "The foreground latent image to composite onto the destination latent."
    assert inputs["x"].tooltip == "Horizontal pixel offset (in 8-pixel steps) for positioning the source latent."
    assert inputs["y"].tooltip == "Vertical pixel offset (in 8-pixel steps) for positioning the source latent."
    assert inputs["resize_source"].tooltip == "If enabled, resizes the source latent to match the dimensions of the destination latent."
    assert inputs["mask"].tooltip == "Optional mask specifying blending intensity (1.0 = source, 0.0 = destination)."

    assert schema.outputs[0].tooltip == "The composited latent output."


def test_image_composite_masked_schema():
    schema = ImageCompositeMasked.define_schema()
    assert schema.node_id == "ImageCompositeMasked"
    assert schema.display_name == "Image Composite Masked"
    assert schema.description == "Composites a source image into a destination image using an optional mask and offset coordinates."
    assert "composite image" in schema.search_aliases
    assert "paste image" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["destination"].tooltip == "The background image into which the source image will be composited."
    assert inputs["source"].tooltip == "The foreground image to composite onto the destination image."
    assert inputs["x"].tooltip == "Horizontal pixel offset for positioning the source image."
    assert inputs["y"].tooltip == "Vertical pixel offset for positioning the source image."
    assert inputs["resize_source"].tooltip == "If enabled, resizes the source image to match the dimensions of the destination image."
    assert inputs["mask"].tooltip == "Optional mask specifying blending intensity (1.0 = source, 0.0 = destination)."

    assert schema.outputs[0].tooltip == "The composited image output."


def test_mask_to_image_schema():
    schema = MaskToImage.define_schema()
    assert schema.node_id == "MaskToImage"
    assert schema.display_name == "Convert Mask to Image"
    assert schema.description == "Converts a 2D grayscale mask into an RGB image with identical channel values."
    assert "mask to RGB" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to convert into an RGB image."
    assert schema.outputs[0].tooltip == "The converted 3-channel RGB image."


def test_image_to_mask_schema():
    schema = ImageToMask.define_schema()
    assert schema.node_id == "ImageToMask"
    assert schema.display_name == "Convert Image to Mask"
    assert schema.description == "Extracts a single color channel (red, green, blue, or alpha) from an image to create a mask."
    assert "image to mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["image"].tooltip == "The source image from which to extract a channel."
    assert inputs["channel"].tooltip == "The specific color or alpha channel to extract as a mask."
    assert schema.outputs[0].tooltip == "The extracted 1-channel mask."


def test_image_color_to_mask_schema():
    schema = ImageColorToMask.define_schema()
    assert schema.node_id == "ImageColorToMask"
    assert schema.display_name == "Image Color to Mask"
    assert schema.description == "Generates a binary mask isolating pixels that exactly match a specified RGB hex color value."
    assert "select color" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["image"].tooltip == "The input image to evaluate for matching color pixels."
    assert inputs["color"].tooltip == "The target RGB integer color value to mask (e.g. 0 for black, 16777215 for white)."
    assert schema.outputs[0].tooltip == "The resulting binary mask with 1.0 at matching color pixels and 0.0 elsewhere."


def test_solid_mask_schema():
    schema = SolidMask.define_schema()
    assert schema.node_id == "SolidMask"
    assert schema.display_name == "Solid Mask"
    assert schema.description == "Creates a uniform, single-value mask of specified width and height."
    assert "solid mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["value"].tooltip == "The uniform fill value for the mask (0.0 = completely transparent/black, 1.0 = completely opaque/white)."
    assert inputs["width"].tooltip == "The width of the mask in pixels."
    assert inputs["height"].tooltip == "The height of the mask in pixels."
    assert schema.outputs[0].tooltip == "The generated solid mask tensor."


def test_invert_mask_schema():
    schema = InvertMask.define_schema()
    assert schema.node_id == "InvertMask"
    assert schema.display_name == "Invert Mask"
    assert schema.description == "Inverts a mask by subtracting its values from 1.0 (black becomes white, white becomes black)."
    assert "negate mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to invert."
    assert schema.outputs[0].tooltip == "The inverted mask output."


def test_crop_mask_schema():
    schema = CropMask.define_schema()
    assert schema.node_id == "CropMask"
    assert schema.display_name == "Crop Mask"
    assert schema.description == "Crops a rectangular sub-region from an input mask specified by offset and dimensions."
    assert "crop mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to crop."
    assert inputs["x"].tooltip == "The top-left X horizontal pixel coordinate of the crop region."
    assert inputs["y"].tooltip == "The top-left Y vertical pixel coordinate of the crop region."
    assert inputs["width"].tooltip == "The width of the cropped region in pixels."
    assert inputs["height"].tooltip == "The height of the cropped region in pixels."
    assert schema.outputs[0].tooltip == "The cropped mask region."


def test_mask_composite_schema():
    schema = MaskComposite.define_schema()
    assert schema.node_id == "MaskComposite"
    assert schema.display_name == "Mask Composite"
    assert schema.description == "Combines two masks using mathematical or logical operations (multiply, add, subtract, and, or, xor)."
    assert "merge masks" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["destination"].tooltip == "The base destination mask onto which the source mask will be combined."
    assert inputs["source"].tooltip == "The source mask to combine with the destination mask."
    assert inputs["x"].tooltip == "Horizontal offset for positioning the source mask onto the destination mask."
    assert inputs["y"].tooltip == "Vertical offset for positioning the source mask onto the destination mask."
    assert inputs["operation"].tooltip == "The blending or boolean operation used to combine the source and destination masks."
    assert schema.outputs[0].tooltip == "The combined mask output."


def test_feather_mask_schema():
    schema = FeatherMask.define_schema()
    assert schema.node_id == "FeatherMask"
    assert schema.display_name == "Feather Mask"
    assert schema.description == "Applies linear gradient softening to the borders of a mask along specified edge distances."
    assert "feather mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to feather."
    assert inputs["left"].tooltip == "Number of pixels to feather inward from the left border."
    assert inputs["top"].tooltip == "Number of pixels to feather inward from the top border."
    assert inputs["right"].tooltip == "Number of pixels to feather inward from the right border."
    assert inputs["bottom"].tooltip == "Number of pixels to feather inward from the bottom border."
    assert schema.outputs[0].tooltip == "The edge-feathered mask output."


def test_grow_mask_schema():
    schema = GrowMask.define_schema()
    assert schema.node_id == "GrowMask"
    assert schema.display_name == "Grow Mask"
    assert schema.description == "Expands (dilates) or shrinks (erodes) mask boundaries by a specified pixel distance."
    assert "dilate mask" in schema.search_aliases
    assert "erode mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to expand or shrink."
    assert inputs["expand"].tooltip == "Number of pixels to grow (positive values expand/dilate, negative values shrink/erode)."
    assert inputs["tapered_corners"].tooltip == "If enabled, uses a cross/diamond footprint for rounded corners; if disabled, uses a full 3x3 square kernel."
    assert schema.outputs[0].tooltip == "The expanded or contracted mask output."


def test_threshold_mask_schema():
    schema = ThresholdMask.define_schema()
    assert schema.node_id == "ThresholdMask"
    assert schema.display_name == "Threshold Mask"
    assert schema.description == "Converts a mask into a binary (0.0 or 1.0) mask based on a cutoff threshold value."
    assert "threshold mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The input mask to threshold."
    assert inputs["value"].tooltip == "The threshold cutoff value. Mask pixels strictly greater than this value become 1.0, others become 0.0."
    assert schema.outputs[0].tooltip == "The binary thresholded mask output."


def test_mask_preview_schema():
    schema = MaskPreview.define_schema()
    assert schema.node_id == "MaskPreview"
    assert schema.display_name == "Preview Mask"
    assert schema.description == "Previews the input mask in the UI and saves it as an image to the ComfyUI output directory."
    assert "preview mask" in schema.search_aliases

    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["mask"].tooltip == "The mask to preview and save."
