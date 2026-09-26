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
            search_aliases=["erode", "dilate", "morphology", "top hat", "bottom hat", "morphological operations"],
            display_name="ImageMorphology",
            description="Applies mathematical morphological operations (such as erosion, dilation, opening, closing, and top-hat/bottom-hat filters) to images.",
            category="image/postprocessing",
            inputs=[
                io.Image.Input("image", tooltip="The input image or batch of images to perform morphological operations on."),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                    tooltip="The morphological operation to apply: 'erode' shrinks bright regions, 'dilate' expands bright regions, 'open' removes small bright spots, 'close' fills small dark holes, 'gradient' highlights edges, and 'top_hat'/'bottom_hat' isolate bright/dark details.",
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1, tooltip="The size of the structuring element matrix (odd integer >= 3). Larger kernels yield stronger morphological effects."),
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
            search_aliases=["color space conversion", "yuv", "ycbcr", "split channels", "yuv split"],
            description="Converts an RGB image into individual Y (luminance), U (chrominance blue-difference), and V (chrominance red-difference) color channel components.",
            category="image/batch",
            inputs=[
                io.Image.Input("image", tooltip="The RGB image to convert into YUV color channels."),
            ],
            outputs=[
                io.Image.Output(display_name="Y", tooltip="The Y (luminance/brightness) channel extracted from the input image."),
                io.Image.Output(display_name="U", tooltip="The U (blue-difference chrominance) channel extracted from the input image."),
                io.Image.Output(display_name="V", tooltip="The V (red-difference chrominance) channel extracted from the input image."),
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
            search_aliases=["color space conversion", "yuv", "ycbcr", "combine channels", "yuv merge", "rgb convert"],
            description="Recombines separate Y (luminance), U (chrominance blue-difference), and V (chrominance red-difference) color channels back into an RGB image.",
            category="image/batch",
            inputs=[
                io.Image.Input("Y", tooltip="The Y (luminance/brightness) channel image input."),
                io.Image.Input("U", tooltip="The U (blue-difference chrominance) channel image input."),
                io.Image.Input("V", tooltip="The V (red-difference chrominance) channel image input."),
            ],
            outputs=[
                io.Image.Output(tooltip="The combined RGB output image."),
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
