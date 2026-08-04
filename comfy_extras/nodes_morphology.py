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
            description="Applies mathematical morphology operations (like erosion, dilation, opening, closing, morphological gradient, top-hat, or bottom-hat) to an image using a square kernel of specified size. This is useful for noise reduction, feature extraction, boundary detection, or mask refinement.",
            search_aliases=["erode", "dilate", "opening", "closing", "gradient", "top hat", "bottom hat", "morphology", "mask refinement"],
            category="image/postprocessing",
            inputs=[
                io.Image.Input("image", tooltip="The input image to apply the morphological operation to."),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                    tooltip="The mathematical morphology operation to perform (e.g., 'erode' to shrink bright regions, 'dilate' to expand bright regions, 'open' to remove small bright details, 'close' to fill small holes).",
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1, tooltip="The size of the square structuring element kernel (must be an odd integer >= 3)."),
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
            display_name="Image RGB to YUV",
            description="Converts an RGB image into its individual luminance (Y) and chrominance (U, V) component channels in the YCbCr/YUV color space. This allows independent processing of brightness and color details (e.g., for advanced masking, color correction, or compression/filtering operations).",
            search_aliases=["color space conversion", "rgb to yuv", "ycbcr", "luma", "chroma", "split channels"],
            category="image/batch",
            inputs=[
                io.Image.Input("image", tooltip="The input RGB image to be converted."),
            ],
            outputs=[
                io.Image.Output(display_name="Y", tooltip="The luminance (brightness) component channel of the image."),
                io.Image.Output(display_name="U", tooltip="The blue-difference chrominance component channel of the image."),
                io.Image.Output(display_name="V", tooltip="The red-difference chrominance component channel of the image."),
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
            display_name="Image YUV to RGB",
            description="Reconstructs an RGB image from its individual luminance (Y) and chrominance (U, V) component channels in the YCbCr/YUV color space. This is used to convert processed channels back to the standard RGB space.",
            search_aliases=["color space conversion", "yuv to rgb", "ycbcr", "combine channels"],
            category="image/batch",
            inputs=[
                io.Image.Input("Y", tooltip="The luminance (brightness) component channel."),
                io.Image.Input("U", tooltip="The blue-difference chrominance component channel."),
                io.Image.Input("V", tooltip="The red-difference chrominance component channel."),
            ],
            outputs=[
                io.Image.Output(tooltip="The reconstructed RGB image."),
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

