import numpy as np
import scipy.ndimage
import torch
import comfy.utils
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, UI

import nodes

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[-2], source.shape[-1]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[-1] - left + min(0, x), destination.shape[-2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[..., top:bottom, left:right]

    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination

class LatentCompositeMasked(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LatentCompositeMasked",
            display_name="Latent Composite Masked",
            description="Composites a source latent into a destination latent using an optional mask and offset coordinates.",
            search_aliases=["overlay latent", "layer latent", "paste latent", "inpaint latent", "composite latent"],
            category="latent",
            inputs=[
                IO.Latent.Input("destination", tooltip="The background latent image into which the source latent will be composited."),
                IO.Latent.Input("source", tooltip="The foreground latent image to composite onto the destination latent."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8, tooltip="Horizontal pixel offset (in 8-pixel steps) for positioning the source latent."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8, tooltip="Vertical pixel offset (in 8-pixel steps) for positioning the source latent."),
                IO.Boolean.Input("resize_source", default=False, tooltip="If enabled, resizes the source latent to match the dimensions of the destination latent."),
                IO.Mask.Input("mask", optional=True, tooltip="Optional mask specifying blending intensity (1.0 = source, 0.0 = destination)."),
            ],
            outputs=[IO.Latent.Output(tooltip="The composited latent output.")],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask = None) -> IO.NodeOutput:
        output = destination.copy()
        destination = destination["samples"].clone()
        source = source["samples"]
        output["samples"] = composite(destination, source, x, y, mask, 8, resize_source)
        return IO.NodeOutput(output)

    composite = execute  # TODO: remove


class ImageCompositeMasked(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCompositeMasked",
            display_name="Image Composite Masked",
            description="Composites a source image into a destination image using an optional mask and offset coordinates.",
            search_aliases=["paste image", "overlay", "layer", "composite image", "blend image"],
            category="image",
            inputs=[
                IO.Image.Input("destination", tooltip="The background image into which the source image will be composited."),
                IO.Image.Input("source", tooltip="The foreground image to composite onto the destination image."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Horizontal pixel offset for positioning the source image."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Vertical pixel offset for positioning the source image."),
                IO.Boolean.Input("resize_source", default=False, tooltip="If enabled, resizes the source image to match the dimensions of the destination image."),
                IO.Mask.Input("mask", optional=True, tooltip="Optional mask specifying blending intensity (1.0 = source, 0.0 = destination)."),
            ],
            outputs=[IO.Image.Output(tooltip="The composited image output.")],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask = None) -> IO.NodeOutput:
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return IO.NodeOutput(output)

    composite = execute  # TODO: remove


class MaskToImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskToImage",
            search_aliases=["convert mask", "mask to RGB", "mask to image"],
            display_name="Convert Mask to Image",
            description="Converts a 2D grayscale mask into an RGB image with identical channel values.",
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to convert into an RGB image."),
            ],
            outputs=[IO.Image.Output(tooltip="The converted 3-channel RGB image.")],
        )

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return IO.NodeOutput(result)

    mask_to_image = execute  # TODO: remove


class ImageToMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageToMask",
            search_aliases=["extract channel", "channel to mask", "image to mask"],
            display_name="Convert Image to Mask",
            description="Extracts a single color channel (red, green, blue, or alpha) from an image to create a mask.",
            category="mask",
            inputs=[
                IO.Image.Input("image", tooltip="The source image from which to extract a channel."),
                IO.Combo.Input("channel", options=["red", "green", "blue", "alpha"], tooltip="The specific color or alpha channel to extract as a mask."),
            ],
            outputs=[IO.Mask.Output(tooltip="The extracted 1-channel mask.")],
        )

    @classmethod
    def execute(cls, image, channel) -> IO.NodeOutput:
        channels = ["red", "green", "blue", "alpha"]
        mask = image[:, :, :, channels.index(channel)]
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove


class ImageColorToMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageColorToMask",
            display_name="Image Color to Mask",
            description="Generates a binary mask isolating pixels that exactly match a specified RGB hex color value.",
            search_aliases=["color keying", "chroma key", "color mask", "select color"],
            category="mask",
            inputs=[
                IO.Image.Input("image", tooltip="The input image to evaluate for matching color pixels."),
                IO.Int.Input("color", default=0, min=0, max=0xFFFFFF, step=1, display_mode=IO.NumberDisplay.number, tooltip="The target RGB integer color value to mask (e.g. 0 for black, 16777215 for white)."),
            ],
            outputs=[IO.Mask.Output(tooltip="The resulting binary mask with 1.0 at matching color pixels and 0.0 elsewhere.")],
        )

    @classmethod
    def execute(cls, image, color) -> IO.NodeOutput:
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]
        mask = torch.where(temp == color, 1.0, 0).float()
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove


class SolidMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SolidMask",
            display_name="Solid Mask",
            description="Creates a uniform, single-value mask of specified width and height.",
            search_aliases=["solid mask", "constant mask", "blank mask", "fill mask"],
            category="mask",
            inputs=[
                IO.Float.Input("value", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="The uniform fill value for the mask (0.0 = completely transparent/black, 1.0 = completely opaque/white)."),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="The width of the mask in pixels."),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="The height of the mask in pixels."),
            ],
            outputs=[IO.Mask.Output(tooltip="The generated solid mask tensor.")],
        )

    @classmethod
    def execute(cls, value, width, height) -> IO.NodeOutput:
        out = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
        return IO.NodeOutput(out)

    solid = execute  # TODO: remove


class InvertMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="InvertMask",
            display_name="Invert Mask",
            description="Inverts a mask by subtracting its values from 1.0 (black becomes white, white becomes black).",
            search_aliases=["reverse mask", "flip mask", "invert mask", "negate mask"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to invert."),
            ],
            outputs=[IO.Mask.Output(tooltip="The inverted mask output.")],
        )

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        out = 1.0 - mask
        return IO.NodeOutput(out)

    invert = execute  # TODO: remove


class CropMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CropMask",
            display_name="Crop Mask",
            description="Crops a rectangular sub-region from an input mask specified by offset and dimensions.",
            search_aliases=["cut mask", "extract mask region", "mask slice", "crop mask"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to crop."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="The top-left X horizontal pixel coordinate of the crop region."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="The top-left Y vertical pixel coordinate of the crop region."),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="The width of the cropped region in pixels."),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="The height of the cropped region in pixels."),
            ],
            outputs=[IO.Mask.Output(tooltip="The cropped mask region.")],
        )

    @classmethod
    def execute(cls, mask, x, y, width, height) -> IO.NodeOutput:
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = mask[:, y:y + height, x:x + width]
        return IO.NodeOutput(out)

    crop = execute  # TODO: remove


class MaskComposite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskComposite",
            display_name="Mask Composite",
            description="Combines two masks using mathematical or logical operations (multiply, add, subtract, and, or, xor).",
            search_aliases=["combine masks", "blend masks", "layer masks", "mask composite", "merge masks"],
            category="mask",
            inputs=[
                IO.Mask.Input("destination", tooltip="The base destination mask onto which the source mask will be combined."),
                IO.Mask.Input("source", tooltip="The source mask to combine with the destination mask."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Horizontal offset for positioning the source mask onto the destination mask."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Vertical offset for positioning the source mask onto the destination mask."),
                IO.Combo.Input("operation", options=["multiply", "add", "subtract", "and", "or", "xor"], tooltip="The blending or boolean operation used to combine the source and destination masks."),
            ],
            outputs=[IO.Mask.Output(tooltip="The combined mask output.")],
        )

    @classmethod
    def execute(cls, destination, source, x, y, operation) -> IO.NodeOutput:
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))

        left, top = (x, y,)
        right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = output[:, top:bottom, left:right]

        if operation == "multiply":
            output[:, top:bottom, left:right] = destination_portion * source_portion
        elif operation == "add":
            output[:, top:bottom, left:right] = destination_portion + source_portion
        elif operation == "subtract":
            output[:, top:bottom, left:right] = destination_portion - source_portion
        elif operation == "and":
            output[:, top:bottom, left:right] = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "or":
            output[:, top:bottom, left:right] = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "xor":
            output[:, top:bottom, left:right] = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()

        output = torch.clamp(output, 0.0, 1.0)

        return IO.NodeOutput(output)

    combine = execute  # TODO: remove


class FeatherMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FeatherMask",
            display_name="Feather Mask",
            description="Applies linear gradient softening to the borders of a mask along specified edge distances.",
            search_aliases=["soft edge mask", "blur mask edges", "gradient mask edge", "feather mask"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to feather."),
                IO.Int.Input("left", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of pixels to feather inward from the left border."),
                IO.Int.Input("top", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of pixels to feather inward from the top border."),
                IO.Int.Input("right", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of pixels to feather inward from the right border."),
                IO.Int.Input("bottom", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of pixels to feather inward from the bottom border."),
            ],
            outputs=[IO.Mask.Output(tooltip="The edge-feathered mask output.")],
        )

    @classmethod
    def execute(cls, mask, left, top, right, bottom) -> IO.NodeOutput:
        output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()

        left = min(left, output.shape[-1])
        right = min(right, output.shape[-1])
        top = min(top, output.shape[-2])
        bottom = min(bottom, output.shape[-2])

        for x in range(left):
            feather_rate = (x + 1.0) / left
            output[:, :, x] *= feather_rate

        for x in range(right):
            feather_rate = (x + 1) / right
            output[:, :, -x] *= feather_rate

        for y in range(top):
            feather_rate = (y + 1) / top
            output[:, y, :] *= feather_rate

        for y in range(bottom):
            feather_rate = (y + 1) / bottom
            output[:, -y, :] *= feather_rate

        return IO.NodeOutput(output)

    feather = execute  # TODO: remove


class GrowMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrowMask",
            search_aliases=["expand mask", "shrink mask", "dilate mask", "erode mask", "grow mask"],
            display_name="Grow Mask",
            description="Expands (dilates) or shrinks (erodes) mask boundaries by a specified pixel distance.",
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to expand or shrink."),
                IO.Int.Input("expand", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1, tooltip="Number of pixels to grow (positive values expand/dilate, negative values shrink/erode)."),
                IO.Boolean.Input("tapered_corners", default=True, advanced=True, tooltip="If enabled, uses a cross/diamond footprint for rounded corners; if disabled, uses a full 3x3 square kernel."),
            ],
            outputs=[IO.Mask.Output(tooltip="The expanded or contracted mask output.")],
        )

    @classmethod
    def execute(cls, mask, expand, tapered_corners) -> IO.NodeOutput:
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return IO.NodeOutput(torch.stack(out, dim=0))

    expand_mask = execute  # TODO: remove


class ThresholdMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ThresholdMask",
            display_name="Threshold Mask",
            description="Converts a mask into a binary (0.0 or 1.0) mask based on a cutoff threshold value.",
            search_aliases=["binary mask", "threshold mask", "binarize mask", "mask cutoff"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask to threshold."),
                IO.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="The threshold cutoff value. Mask pixels strictly greater than this value become 1.0, others become 0.0."),
            ],
            outputs=[IO.Mask.Output(tooltip="The binary thresholded mask output.")],
        )

    @classmethod
    def execute(cls, mask, value) -> IO.NodeOutput:
        mask = (mask > value).float()
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove


# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class MaskPreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskPreview",
            search_aliases=["show mask", "view mask", "inspect mask", "debug mask", "preview mask"],
            display_name="Preview Mask",
            category="mask",
            description="Previews the input mask in the UI and saves it as an image to the ComfyUI output directory.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The mask to preview and save."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mask, filename_prefix="ComfyUI") -> IO.NodeOutput:
        return IO.NodeOutput(ui=UI.PreviewMask(mask))


class MaskExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
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
        ]


async def comfy_entrypoint() -> MaskExtension:
    return MaskExtension()
