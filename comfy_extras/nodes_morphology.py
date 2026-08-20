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
            display_name="ImageMorphology",
            description="Apply mathematical morphology operations (such as erosion, dilation, opening, closing, or top hat) to process image structures and mask shapes.",
            search_aliases=["erode", "dilate", "morphology", "opening", "closing", "top hat", "bottom hat", "kernel"],
            category="image/postprocessing",
            inputs=[
                io.Image.Input("image", tooltip="The input image or mask to process with mathematical morphology."),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                    tooltip="Morphological operation: 'erode' shrinks features, 'dilate' expands them, 'open' removes small noise, 'close' fills small gaps, 'gradient' highlights edges, 'top_hat' isolates bright details, 'bottom_hat' isolates dark details.",
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1, tooltip="Size of the structuring element kernel in pixels (must be an odd integer, e.g. 3, 5, 7)."),
            ],
            outputs=[
                io.Image.Output(id="IMAGE", display_name="IMAGE", tooltip="The morphologically filtered output image."),
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
            display_name="RGB to YUV",
            description="Convert an RGB image into separate Y (Luminance), U (Chrominance-Blue), and V (Chrominance-Red) color channels.",
            search_aliases=["color space conversion", "rgb to yuv", "yuv split", "luminance", "chrominance"],
            category="image/batch",
            inputs=[
                io.Image.Input("image", tooltip="The input RGB image to convert into YUV color channels."),
            ],
            outputs=[
                io.Image.Output(id="Y", display_name="Y", tooltip="Luminance (brightness) channel as a grayscale image."),
                io.Image.Output(id="U", display_name="U", tooltip="U chrominance (blue-difference) channel as a grayscale image."),
                io.Image.Output(id="V", display_name="V", tooltip="V chrominance (red-difference) channel as a grayscale image."),
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
            display_name="YUV to RGB",
            description="Recombine separate Y (Luminance), U (Chrominance-Blue), and V (Chrominance-Red) color channels into a single RGB image.",
            search_aliases=["color space conversion", "yuv to rgb", "yuv merge", "luminance", "chrominance"],
            category="image/batch",
            inputs=[
                io.Image.Input("Y", tooltip="Luminance (brightness) channel image."),
                io.Image.Input("U", tooltip="U chrominance (blue-difference) channel image."),
                io.Image.Input("V", tooltip="V chrominance (red-difference) channel image."),
            ],
            outputs=[
                io.Image.Output(id="IMAGE", display_name="IMAGE", tooltip="Recombined RGB image."),
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

