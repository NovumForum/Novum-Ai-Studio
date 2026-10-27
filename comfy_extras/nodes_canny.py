from kornia.filters import canny
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io


class Canny(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Canny",
            display_name="Canny",
            description="Detects structural boundaries and outlines in images using the Canny edge detection algorithm.",
            search_aliases=["edge detection", "outline", "contour detection", "line art"],
            category="image/preprocessors",
            essentials_category="Image Tools",
            inputs=[
                io.Image.Input("image", tooltip="The input image to process for edge detection."),
                io.Float.Input("low_threshold", default=0.4, min=0.01, max=0.99, step=0.01, tooltip="Lower bound threshold for hysteresis edge tracing. Edges with gradient intensity below this value are discarded."),
                io.Float.Input("high_threshold", default=0.8, min=0.01, max=0.99, step=0.01, tooltip="Upper bound threshold for hysteresis edge tracing. Edges with gradient intensity above this value are preserved."),
            ],
            outputs=[io.Image.Output(tooltip="The output black-and-white edge map image.")],
        )

    @classmethod
    def detect_edge(cls, image, low_threshold, high_threshold):
        # Deprecated: use the V3 schema's `execute` method instead of this.
        return cls.execute(image, low_threshold, high_threshold)

    @classmethod
    def execute(cls, image, low_threshold, high_threshold) -> io.NodeOutput:
        output = canny(image.to(comfy.model_management.get_torch_device()).movedim(-1, 1), low_threshold, high_threshold)
        img_out = output[1].to(comfy.model_management.intermediate_device()).repeat(1, 3, 1, 1).movedim(1, -1)
        return io.NodeOutput(img_out)


class CannyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Canny]


async def comfy_entrypoint() -> CannyExtension:
    return CannyExtension()
