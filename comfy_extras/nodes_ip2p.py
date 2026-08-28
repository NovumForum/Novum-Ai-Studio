import torch

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class InstructPixToPixConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InstructPixToPixConditioning",
            display_name="Instruct PixToPix Conditioning",
            category="conditioning/instructpix2pix",
            description="Prepares conditioning inputs for InstructPixToPix image editing models by combining positive/negative prompts with target image latents.",
            search_aliases=["instructpix2pix", "ip2p", "image editing", "instruct conditioning", "pix2pix"],
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive prompt conditioning to guide the edit."),
                io.Conditioning.Input("negative", tooltip="Negative prompt conditioning to guide what to avoid during the edit."),
                io.Vae.Input("vae", tooltip="VAE model used to encode the reference image pixels into latent space."),
                io.Image.Input("pixels", tooltip="Input reference image to be edited by InstructPixToPix."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="Updated positive conditioning containing concatenated latent image representations."),
                io.Conditioning.Output(display_name="negative", tooltip="Updated negative conditioning containing concatenated latent image representations."),
                io.Latent.Output(display_name="latent", tooltip="Empty target latent canvas initialized with matching spatial dimensions."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, pixels, vae) -> io.NodeOutput:
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8

        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]

        concat_latent = vae.encode(pixels)

        out_latent = {}
        out_latent["samples"] = torch.zeros_like(concat_latent)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return io.NodeOutput(out[0], out[1], out_latent)


class InstructPix2PixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            InstructPixToPixConditioning,
        ]


async def comfy_entrypoint() -> InstructPix2PixExtension:
    return InstructPix2PixExtension()

