from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ColorToRGBInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorToRGBInt",
            display_name="Color to RGB Int",
            category="utils",
            description="Convert a hex color string (#RRGGBB) to a single RGB integer value.",
            search_aliases=["hex to int", "color picker", "color to integer", "hex color", "rgb integer"],
            inputs=[
                io.Color.Input("color", tooltip="Hex color code in #RRGGBB format."),
            ],
            outputs=[
                io.Int.Output(display_name="rgb_int", tooltip="Integer representation of the RGB color value."),
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
