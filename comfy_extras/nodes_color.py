from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ColorToRGBInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorToRGBInt",
            display_name="Color to RGB Int",
            category="utils",
            search_aliases=["hex to rgb", "color code", "color converter", "rgb int", "hex to int"],
            description="Convert a color to a RGB integer value.",
            inputs=[
                io.Color.Input(
                    "color",
                    tooltip="The hex color code (e.g. #RRGGBB) to convert into an integer RGB value.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    id="rgb_int",
                    display_name="rgb_int",
                    tooltip="The converted integer RGB value representing the color.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        color: str,
    ) -> io.NodeOutput:
        # expect format #RRGGBB
        if len(color) != 7 or color[0] != "#":
            raise ValueError("Color must be in format #RRGGBB")
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return io.NodeOutput(r * 256 * 256 + g * 256 + b)


class ColorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ColorToRGBInt]


async def comfy_entrypoint() -> ColorExtension:
    return ColorExtension()
