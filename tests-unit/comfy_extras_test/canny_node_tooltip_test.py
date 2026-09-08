import comfy.cli_args
comfy.cli_args.args.cpu = True

import pytest
import torch
from comfy_extras.nodes_canny import Canny


def test_canny_schema_metadata_and_tooltips():
    schema = Canny.define_schema()
    assert schema.node_id == "Canny"
    assert schema.display_name == "Canny"
    assert "Canny edge detection algorithm" in schema.description
    assert "edge detection" in schema.search_aliases
    assert "line art" in schema.search_aliases

    # Inputs
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_by_id
    assert "low_threshold" in inputs_by_id
    assert "high_threshold" in inputs_by_id

    assert "perform edge detection" in inputs_by_id["image"].tooltip
    assert "Lower bound intensity threshold" in inputs_by_id["low_threshold"].tooltip
    assert "Upper bound intensity threshold" in inputs_by_id["high_threshold"].tooltip

    # Outputs
    assert len(schema.outputs) == 1
    assert "binary black-and-white edge map" in schema.outputs[0].tooltip


def test_canny_execution():
    dummy_image = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    output = Canny.execute(image=dummy_image, low_threshold=0.4, high_threshold=0.8)
    assert output is not None
    res_tensor = output.result[0]
    assert res_tensor.shape == (1, 32, 32, 3)
