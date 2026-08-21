from comfy.cli_args import args
args.cpu = True

import pytest
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel


def test_upscale_model_loader_schema():
    schema = UpscaleModelLoader.define_schema()
    assert schema.node_id == "UpscaleModelLoader"
    assert schema.display_name == "Load Upscale Model"
    assert schema.description is not None
    assert "super-resolution" in schema.description
    assert schema.search_aliases is not None
    assert "esrgan" in schema.search_aliases
    assert "swinir" in schema.search_aliases

    # Check input tooltips
    input_dict = {inp.id: inp for inp in schema.inputs}
    assert "model_name" in input_dict
    assert input_dict["model_name"].tooltip is not None
    assert "models/upscale_models" in input_dict["model_name"].tooltip

    # Check output tooltips
    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None
    assert "descriptor" in schema.outputs[0].tooltip


def test_image_upscale_with_model_schema():
    schema = ImageUpscaleWithModel.define_schema()
    assert schema.node_id == "ImageUpscaleWithModel"
    assert schema.display_name == "Upscale Image (using Model)"
    assert schema.description is not None
    assert "tiled processing" in schema.description
    assert schema.search_aliases is not None
    assert "super resolution" in schema.search_aliases
    assert "esrgan" in schema.search_aliases

    # Check input tooltips
    input_dict = {inp.id: inp for inp in schema.inputs}
    assert "upscale_model" in input_dict
    assert input_dict["upscale_model"].tooltip is not None
    assert "loaded upscale model" in input_dict["upscale_model"].tooltip

    assert "image" in input_dict
    assert input_dict["image"].tooltip is not None
    assert "image tensor" in input_dict["image"].tooltip

    # Check output tooltips
    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None
    assert "upscaled output image" in schema.outputs[0].tooltip
