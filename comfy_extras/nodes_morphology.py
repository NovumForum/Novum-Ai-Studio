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
            display_name="Image Morphology",
            description="Apply mathematical morphology operations (erosion, dilation, opening, closing, morphological gradient, top-hat, bottom-hat) to an image using a square structuring element.",
            search_aliases=["morphology", "erode", "dilate", "opening", "closing", "gradient", "top hat", "bottom hat", "structuring element", "mask processing"],
            category="image/postprocessing",
            inputs=[
                io.Image.Input("image", tooltip="The input image to process using morphological operations."),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                    tooltip="The mathematical morphology operation to perform:\n- erode: Shrinks bright regions/expands dark regions.\n- dilate: Expands bright regions/shrinks dark regions.\n- open: Erosion followed by dilation (removes small bright spots).\n- close: Dilation followed by erosion (fills small dark holes).\n- gradient: Difference between dilation and erosion (highlights edges).\n- top_hat: Difference between original image and its opening.\n- bottom_hat: Difference between closing and original image.",
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1, tooltip="The width and height of the square kernel structuring element (must be an odd integer >= 3)."),
            ],
            outputs=[
                io.Image.Output(tooltip="The morphologically processed image."),
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
            display_name="RGB to YUV (YCbCr)",
            description="Convert an RGB image into separate Y (luma), U (chrominance blue-difference), and V (chrominance red-difference) color space channels.",
            search_aliases=["color space conversion", "rgb to yuv", "rgb to ycbcr", "luma", "chroma", "separate channels"],
            category="image/batch",
            inputs=[
                io.Image.Input("image", tooltip="The RGB image to convert into YUV / YCbCr channels."),
            ],
            outputs=[
                io.Image.Output(display_name="Y", tooltip="Luminance (luma / brightness) channel."),
                io.Image.Output(display_name="U", tooltip="Chroma blue-difference channel."),
                io.Image.Output(display_name="V", tooltip="Chroma red-difference channel."),
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
            display_name="YUV (YCbCr) to RGB",
            description="Combine separate Y (luma), U (chrominance blue-difference), and V (chrominance red-difference) channels back into a single RGB image.",
            search_aliases=["color space conversion", "yuv to rgb", "ycbcr to rgb", "recombine channels", "merge channels"],
            category="image/batch",
            inputs=[
                io.Image.Input("Y", tooltip="Luminance (luma / brightness) channel image."),
                io.Image.Input("U", tooltip="Chroma blue-difference channel image."),
                io.Image.Input("V", tooltip="Chroma red-difference channel image."),
            ],
            outputs=[
                io.Image.Output(tooltip="The combined RGB image."),
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
