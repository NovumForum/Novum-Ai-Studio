import pytest
import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True
from comfy_extras.nodes_mask import SolidMask, InvertMask, ThresholdMask


def test_solid_mask_schema():
    schema = SolidMask.define_schema()
    assert schema.node_id == "SolidMask"
    assert schema.display_name == "Solid Mask"
    assert schema.category == "mask"
    assert "solid uniform mask" in schema.description
    assert "create mask" in schema.search_aliases
    assert "uniform mask" in schema.search_aliases

    input_names = [inp.id for inp in schema.inputs]
    assert "value" in input_names
    assert "width" in input_names
    assert "height" in input_names

    val_inp = next(inp for inp in schema.inputs if inp.id == "value")
    assert val_inp.tooltip == "Fill value for the mask (0.0 = unmasked/black, 1.0 = fully masked/white)."

    out = schema.outputs[0]
    assert out.tooltip == "The generated solid mask tensor."


def test_solid_mask_execution():
    res = SolidMask.execute(value=0.5, width=16, height=16)
    tensor = res.args[0]
    assert tensor.shape == (1, 16, 16)
    assert torch.allclose(tensor, torch.tensor(0.5))


def test_invert_mask_schema():
    schema = InvertMask.define_schema()
    assert schema.node_id == "InvertMask"
    assert schema.display_name == "Invert Mask"
    assert schema.category == "mask"
    assert "Inverts a mask" in schema.description
    assert "reverse mask" in schema.search_aliases
    assert "negate mask" in schema.search_aliases

    mask_inp = schema.inputs[0]
    assert mask_inp.id == "mask"
    assert mask_inp.tooltip == "The input mask to invert."

    out = schema.outputs[0]
    assert out.tooltip == "The inverted mask tensor."


def test_invert_mask_execution():
    mask = torch.tensor([[[0.0, 0.2], [0.8, 1.0]]])
    res = InvertMask.execute(mask=mask)
    tensor = res.args[0]
    expected = torch.tensor([[[1.0, 0.8], [0.2, 0.0]]])
    assert torch.allclose(tensor, expected)


def test_threshold_mask_schema():
    schema = ThresholdMask.define_schema()
    assert schema.node_id == "ThresholdMask"
    assert schema.display_name == "Threshold Mask"
    assert schema.category == "mask"
    assert "Converts a mask to binary" in schema.description
    assert "binary mask" in schema.search_aliases
    assert "binarize mask" in schema.search_aliases

    mask_inp = next(inp for inp in schema.inputs if inp.id == "mask")
    assert mask_inp.tooltip == "The input mask to threshold."

    val_inp = next(inp for inp in schema.inputs if inp.id == "value")
    assert val_inp.tooltip == "Cutoff threshold. Values strictly greater than this become 1.0, others 0.0."

    out = schema.outputs[0]
    assert out.tooltip == "The binarized output mask tensor."


def test_threshold_mask_execution():
    mask = torch.tensor([[[0.2, 0.5], [0.51, 0.9]]])
    res = ThresholdMask.execute(mask=mask, value=0.5)
    tensor = res.args[0]
    expected = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    assert torch.allclose(tensor, expected)
