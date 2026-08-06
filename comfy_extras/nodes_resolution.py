from __future__ import annotations
import math
from enum import Enum
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_H = "4:3 (Standard)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"
    PHOTO_V = "2:3 (Portrait Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.WIDESCREEN_V: (9, 16),
}


class ResolutionSelector(io.ComfyNode):
    """Calculate width and height from aspect ratio and megapixel target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResolutionSelector",
            display_name="Resolution Selector",
            category="utils",
            search_aliases=[
                "aspect ratio",
                "aspect",
                "dimensions",
                "resolution",
                "image size",
                "latent size",
                "megapixels",
                "empty latent size",
                "width",
                "height",
                "ratio",
            ],
            description="Automatically calculate the optimal width and height dimensions in pixels (rounded to multiples of 8) based on a target aspect ratio and total megapixels. Perfect for dynamically setting the size of Empty Latent Images for Stable Diffusion, Flux, or upscale pipelines without manually doing the math.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="Choose from standard aspect ratios like 1:1, 16:9, or portrait formats. Calculates dimensions keeping this ratio.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target total pixel count in megapixels (e.g. 1.0 MP is 1024x1024, 0.25 MP is 512x512). Higher values generate larger, higher-fidelity images but require more VRAM.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "width",
                    tooltip="Calculated target width in pixels, rounded to the nearest multiple of 8 for optimal neural network processing.",
                ),
                io.Int.Output(
                    "height",
                    tooltip="Calculated target height in pixels, rounded to the nearest multiple of 8 for optimal neural network processing.",
                ),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, megapixels: float) -> io.NodeOutput:
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / 8) * 8
        height = round(h_ratio * scale / 8) * 8
        return io.NodeOutput(width, height)


class ResolutionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ResolutionSelector,
        ]


async def comfy_entrypoint() -> ResolutionExtension:
    return ResolutionExtension()
