from typing_extensions import override

import nodes
from comfy_api.latest import ComfyExtension, io


class CLIPTextEncodeSDXLRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeSDXLRefiner",
            display_name="CLIP Text Encode (SDXL Refiner)",
            description="Encodes prompt text into conditioning specifically formatted for the SDXL Refiner model, incorporating aesthetic score and image dimensions.",
            search_aliases=["sdxl refiner prompt", "refiner clip encode", "aesthetic prompt", "sdxl prompt refiner"],
            category="advanced/conditioning",
            inputs=[
                io.Float.Input("ascore", default=6.0, min=0.0, max=1000.0, step=0.01, tooltip="Target aesthetic score for the refiner model (default 6.0)."),
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Target image width for model conditioning."),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Target image height for model conditioning."),
                io.String.Input("text", multiline=True, dynamic_prompts=True, tooltip="Text prompt to encode for the refiner model."),
                io.Clip.Input("clip", tooltip="The CLIP model used for text encoding."),
            ],
            outputs=[io.Conditioning.Output(tooltip="Conditioning tensor containing encoded prompt embeddings with aesthetic score and dimension metadata for the SDXL refiner.")],
        )

    @classmethod
    def execute(cls, clip, ascore, width, height, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"aesthetic_score": ascore, "width": width, "height": height}))

class CLIPTextEncodeSDXL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeSDXL",
            display_name="CLIP Text Encode (SDXL)",
            description="Encodes dual text prompts (global text_g and local text_l) into conditioning formatted for SDXL, incorporating crop and target resolution metadata.",
            search_aliases=["sdxl prompt", "dual clip encode", "sdxl text encode", "clip text encode sdxl"],
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip", tooltip="The CLIP model used for text encoding."),
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Original canvas width for conditioning metadata."),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Original canvas height for conditioning metadata."),
                io.Int.Input("crop_w", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True, tooltip="Left crop coordinate offset for conditioning metadata."),
                io.Int.Input("crop_h", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True, tooltip="Top crop coordinate offset for conditioning metadata."),
                io.Int.Input("target_width", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Target resolution width for conditioning metadata."),
                io.Int.Input("target_height", default=1024, min=0, max=nodes.MAX_RESOLUTION, tooltip="Target resolution height for conditioning metadata."),
                io.String.Input("text_g", multiline=True, dynamic_prompts=True, tooltip="Global text prompt (CLIP G/L combined embedding)."),
                io.String.Input("text_l", multiline=True, dynamic_prompts=True, tooltip="Local text prompt (openCLIP G model embedding)."),
            ],
            outputs=[io.Conditioning.Output(tooltip="Conditioning tensor containing dual-encoded prompt embeddings with SDXL resolution and crop metadata.")],
        )

    @classmethod
    def execute(cls, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l) -> io.NodeOutput:
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}))


class ClipSdxlExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeSDXLRefiner,
            CLIPTextEncodeSDXL,
        ]


async def comfy_entrypoint() -> ClipSdxlExtension:
    return ClipSdxlExtension()
