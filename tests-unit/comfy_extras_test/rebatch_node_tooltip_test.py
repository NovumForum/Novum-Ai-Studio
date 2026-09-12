import torch
import pytest
from comfy_extras.nodes_rebatch import LatentRebatch, ImageRebatch


def test_latent_rebatch_schema():
    schema = LatentRebatch.define_schema()
    assert schema.node_id == "RebatchLatents"
    assert schema.display_name == "Rebatch Latents"
    assert "Rebatches a list of latent dictionaries" in schema.description
    assert "rebatch latents" in schema.search_aliases
    assert "batch latents" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert "latents" in input_map
    assert input_map["latents"].tooltip == "The list of latent objects to rebatch."
    assert "batch_size" in input_map
    assert input_map["batch_size"].tooltip == "Target batch size for each output latent batch."

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "List of rebatched latent objects."


def test_image_rebatch_schema():
    schema = ImageRebatch.define_schema()
    assert schema.node_id == "RebatchImages"
    assert schema.display_name == "Rebatch Images"
    assert "Rebatches a list of image tensors" in schema.description
    assert "rebatch images" in schema.search_aliases
    assert "batch images" in schema.search_aliases

    input_map = {inp.id: inp for inp in schema.inputs}
    assert "images" in input_map
    assert input_map["images"].tooltip == "The list of image tensors to rebatch."
    assert "batch_size" in input_map
    assert input_map["batch_size"].tooltip == "Target batch size for each output image batch."

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "List of rebatched image tensors."


def test_image_rebatch_execution():
    img1 = torch.zeros((2, 64, 64, 3))
    img2 = torch.ones((3, 64, 64, 3))
    out = ImageRebatch.execute([img1, img2], [2])
    output_list = out.args[0]
    assert len(output_list) == 3  # 5 items rebatched into 2 + 2 + 1
    assert output_list[0].shape[0] == 2
    assert output_list[1].shape[0] == 2
    assert output_list[2].shape[0] == 1


def test_latent_rebatch_execution():
    lat1 = {"samples": torch.zeros((2, 4, 32, 32))}
    lat2 = {"samples": torch.ones((3, 4, 32, 32))}
    out = LatentRebatch.execute([lat1, lat2], [2])
    output_list = out.args[0]
    assert len(output_list) == 3  # 5 items rebatched into 2 + 2 + 1
    assert output_list[0]["samples"].shape[0] == 2
    assert output_list[1]["samples"].shape[0] == 2
    assert output_list[2]["samples"].shape[0] == 1
