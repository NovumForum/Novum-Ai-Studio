import torch
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from kornia.morphology import dilation, erosion, opening, closing, gradient, top_hat, bottom_hat
import kornia.color


class Morphology(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Morphology",
            search_aliases=["morphology", "erode", "dilate", "opening", "closing", "top hat", "bottom hat"],
            display_name="ImageMorphology",
            category="image/postprocessing",
            description="Applies mathematical morphology operations (erode, dilate, open, close, gradient, top hat, bottom hat) to image tensors.",
            inputs=[
                io.Image.Input("image", tooltip="The input image tensor."),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                    tooltip="The morphology operation: 'erode' shrinks bright regions, 'dilate' expands bright regions, 'open' removes small noise, 'close' fills small holes, 'gradient' highlights boundaries, 'top_hat' extracts bright details, 'bottom_hat' extracts dark details.",
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1, tooltip="The structuring element kernel size in pixels (must be odd)."),
            ],
            outputs=[
                io.Image.Output(tooltip="The image tensor after applying the morphological operation."),
            ],
        )

    @classmethod
    def execute(cls, image, operation, kernel_size) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        kernel = torch.ones(kernel_size, kernel_size, device=device)
        image_k = image.to(device).movedim(-1, 1)
        if operation == "erode":
            output = erosion(image_k, kernel)
        elif operation == "dilate":
            output = dilation(image_k, kernel)
        elif operation == "open":
            output = opening(image_k, kernel)
        elif operation == "close":
            output = closing(image_k, kernel)
        elif operation == "gradient":
            output = gradient(image_k, kernel)
        elif operation == "top_hat":
            output = top_hat(image_k, kernel)
        elif operation == "bottom_hat":
            output = bottom_hat(image_k, kernel)
        else:
            raise ValueError(f"Invalid operation {operation} for morphology. Must be one of 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'")
        img_out = output.to(comfy.model_management.intermediate_device()).movedim(1, -1)
        return io.NodeOutput(img_out)


class ImageRGBToYUV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageRGBToYUV",
            search_aliases=["rgb to yuv", "ycbcr", "color space conversion", "yuv split", "separate channels"],
            category="image/batch",
            description="Converts an RGB image tensor into separate Y (Luminance), U (Chroma blue-difference), and V (Chroma red-difference) channel images.",
            inputs=[
                io.Image.Input("image", tooltip="The input RGB image tensor."),
            ],
            outputs=[
                io.Image.Output(display_name="Y", tooltip="The Luminance (Y) channel image tensor."),
                io.Image.Output(display_name="U", tooltip="The Chroma blue-difference (U) channel image tensor."),
                io.Image.Output(display_name="V", tooltip="The Chroma red-difference (V) channel image tensor."),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        out = kornia.color.rgb_to_ycbcr(image.movedim(-1, 1)).movedim(1, -1)
        return io.NodeOutput(out[..., 0:1].expand_as(image), out[..., 1:2].expand_as(image), out[..., 2:3].expand_as(image))

class ImageYUVToRGB(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageYUVToRGB",
            search_aliases=["yuv to rgb", "ycbcr", "color space conversion", "yuv merge", "combine channels"],
            category="image/batch",
            description="Reconstructs an RGB image tensor from separate Y (Luminance), U (Chroma blue-difference), and V (Chroma red-difference) channel images.",
            inputs=[
                io.Image.Input("Y", tooltip="The Luminance (Y) channel image tensor."),
                io.Image.Input("U", tooltip="The Chroma blue-difference (U) channel image tensor."),
                io.Image.Input("V", tooltip="The Chroma red-difference (V) channel image tensor."),
            ],
            outputs=[
                io.Image.Output(tooltip="The reconstructed RGB image tensor."),
            ],
        )

    @classmethod
    def execute(cls, Y, U, V) -> io.NodeOutput:
        image = torch.cat([torch.mean(Y, dim=-1, keepdim=True), torch.mean(U, dim=-1, keepdim=True), torch.mean(V, dim=-1, keepdim=True)], dim=-1)
        out = kornia.color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1)
        return io.NodeOutput(out)


class MorphologyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Morphology,
            ImageRGBToYUV,
            ImageYUVToRGB,
        ]


async def comfy_entrypoint() -> MorphologyExtension:
    return MorphologyExtension()

