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
            search_aliases=["overlay latent", "layer latent", "paste latent", "inpaint latent"],
            category="latent",
            description="Composites a source latent into a destination latent batch at specified offsets, optionally masked.",
            inputs=[
                IO.Latent.Input("destination", tooltip="The destination latent batch to receive the composite operation."),
                IO.Latent.Input("source", tooltip="The source latent batch to paste onto the destination."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8, tooltip="X-offset position in pixels for overlaying the source latent."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8, tooltip="Y-offset position in pixels for overlaying the source latent."),
                IO.Boolean.Input("resize_source", default=False, tooltip="When enabled, resizes source latent to match destination dimensions."),
                IO.Mask.Input("mask", optional=True, tooltip="Optional mask specifying blending opacity for compositing source onto destination."),
            ],
            outputs=[IO.Latent.Output(tooltip="The composited latent batch.")],
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
            search_aliases=["paste image", "overlay", "layer"],
            category="image",
            description="Overlays a source image onto a destination image at specified pixel coordinates with optional mask blending.",
            inputs=[
                IO.Image.Input("destination", tooltip="The background destination image tensor."),
                IO.Image.Input("source", tooltip="The foreground source image tensor to overlay."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="X pixel coordinate position for the top-left of the source image."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Y pixel coordinate position for the top-left of the source image."),
                IO.Boolean.Input("resize_source", default=False, tooltip="When enabled, resizes source image to match destination dimensions."),
                IO.Mask.Input("mask", optional=True, tooltip="Optional opacity mask for blending the source image."),
            ],
            outputs=[IO.Image.Output(tooltip="The composited output image tensor.")],
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
            search_aliases=["convert mask"],
            display_name="Convert Mask to Image",
            category="mask",
            description="Converts a single-channel grayscale mask into a 3-channel RGB image tensor.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input single-channel mask tensor."),
            ],
            outputs=[IO.Image.Output(tooltip="The resulting 3-channel RGB image tensor.")],
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
            search_aliases=["extract channel", "channel to mask"],
            display_name="Convert Image to Mask",
            category="mask",
            description="Extracts a specified color or alpha channel from an image into a single-channel mask.",
            inputs=[
                IO.Image.Input("image", tooltip="The input RGB or RGBA image tensor."),
                IO.Combo.Input("channel", options=["red", "green", "blue", "alpha"], tooltip="The specific color or alpha channel to extract as a mask."),
            ],
            outputs=[IO.Mask.Output(tooltip="The extracted single-channel mask tensor.")],
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
            search_aliases=["color keying", "chroma key"],
            category="mask",
            description="Generates a binary mask isolating matching pixels of a target hex color code.",
            inputs=[
                IO.Image.Input("image", tooltip="The input image tensor to perform color keying on."),
                IO.Int.Input("color", default=0, min=0, max=0xFFFFFF, step=1, display_mode=IO.NumberDisplay.number, tooltip="Target RGB color code in integer hex format (e.g. 0x00FF00 for green screen)."),
            ],
            outputs=[IO.Mask.Output(tooltip="The resulting binary mask isolating matching target color pixels.")],
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
            category="mask",
            description="Creates a solid single-channel mask tensor of specified dimensions and uniform value.",
            inputs=[
                IO.Float.Input("value", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Uniform mask fill value ranging from 0.0 (black/unmasked) to 1.0 (white/fully masked)."),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Width of the generated mask tensor in pixels."),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Height of the generated mask tensor in pixels."),
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
            search_aliases=["reverse mask", "flip mask"],
            category="mask",
            description="Inverts a mask, swapping masked (1.0) and unmasked (0.0) regions.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask tensor to invert."),
            ],
            outputs=[IO.Mask.Output(tooltip="The inverted mask tensor.")],
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
            search_aliases=["cut mask", "extract mask region", "mask slice"],
            category="mask",
            description="Crops a rectangular sub-region from an input mask tensor.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask tensor to crop."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Starting X coordinate pixel position for the cropping rectangle."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Starting Y coordinate pixel position for the cropping rectangle."),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Width of the cropped mask region in pixels."),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1, tooltip="Height of the cropped mask region in pixels."),
            ],
            outputs=[IO.Mask.Output(tooltip="The cropped rectangular sub-mask tensor.")],
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
            search_aliases=["combine masks", "blend masks", "layer masks"],
            category="mask",
            description="Combines two masks using arithmetic or logical boolean operations (multiply, add, subtract, and, or, xor).",
            inputs=[
                IO.Mask.Input("destination", tooltip="The base destination mask tensor."),
                IO.Mask.Input("source", tooltip="The source mask tensor to overlay onto destination."),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="X coordinate offset in pixels for placing source onto destination."),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Y coordinate offset in pixels for placing source onto destination."),
                IO.Combo.Input("operation", options=["multiply", "add", "subtract", "and", "or", "xor"], tooltip="The blending or boolean logic operation used to merge masks."),
            ],
            outputs=[IO.Mask.Output(tooltip="The combined composite mask tensor.")],
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
            search_aliases=["soft edge mask", "blur mask edges", "gradient mask edge"],
            category="mask",
            description="Applies a soft linear gradient feather to the outer edges of a mask.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask tensor to feather."),
                IO.Int.Input("left", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Feather distance in pixels for the left border."),
                IO.Int.Input("top", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Feather distance in pixels for the top border."),
                IO.Int.Input("right", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Feather distance in pixels for the right border."),
                IO.Int.Input("bottom", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Feather distance in pixels for the bottom border."),
            ],
            outputs=[IO.Mask.Output(tooltip="The edge-feathered mask tensor.")],
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
            search_aliases=["expand mask", "shrink mask"],
            display_name="Grow Mask",
            category="mask",
            description="Expands (dilates) or shrinks (erodes) a mask by a specified pixel distance.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input mask tensor to expand or shrink."),
                IO.Int.Input("expand", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1, tooltip="Amount in pixels to expand (positive values) or shrink (negative values) the mask."),
                IO.Boolean.Input("tapered_corners", default=True, advanced=True, tooltip="When enabled, uses a rounded/tapered kernel for smoother corner expansion."),
            ],
            outputs=[IO.Mask.Output(tooltip="The expanded or contracted mask tensor.")],
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
            search_aliases=["binary mask"],
            category="mask",
            description="Converts a soft grayscale mask into a binary mask by thresholding values above a cutoff.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The input grayscale mask tensor."),
                IO.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Threshold cutoff value between 0.0 and 1.0."),
            ],
            outputs=[IO.Mask.Output(tooltip="The binary thresholded mask tensor.")],
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
            search_aliases=["show mask", "view mask", "inspect mask", "debug mask"],
            display_name="Preview Mask",
            category="mask",
            description="Previews a mask tensor directly in the ComfyUI interface and saves it to the output directory.",
            inputs=[
                IO.Mask.Input("mask", tooltip="The mask tensor to preview and save."),
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
