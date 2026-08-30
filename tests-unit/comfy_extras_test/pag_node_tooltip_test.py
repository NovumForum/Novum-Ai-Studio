import comfy.cli_args
comfy.cli_args.args.cpu = True

from unittest.mock import MagicMock
from comfy_extras.nodes_pag import PerturbedAttentionGuidance


def test_pag_schema_metadata():
    schema = PerturbedAttentionGuidance.define_schema()

    assert schema.node_id == "PerturbedAttentionGuidance"
    assert schema.display_name == "Perturbed-Attention Guidance (PAG)"
    assert "Applies Perturbed-Attention Guidance" in schema.description

    expected_aliases = {
        "pag",
        "perturbed attention guidance",
        "attention guidance",
        "quality boost",
        "structure enhancement",
    }
    assert expected_aliases.issubset(set(schema.search_aliases))

    # Inputs verification
    input_map = {inp.id: inp for inp in schema.inputs}
    assert "model" in input_map
    assert "scale" in input_map

    assert "The diffusion model" in input_map["model"].tooltip
    assert "Guidance scale" in input_map["scale"].tooltip

    # Outputs verification
    assert len(schema.outputs) == 1
    assert "patched diffusion model" in schema.outputs[0].tooltip


def test_pag_execution():
    mock_model = MagicMock()
    cloned_model = MagicMock()
    mock_model.clone.return_value = cloned_model

    result = PerturbedAttentionGuidance.execute(mock_model, scale=3.0)

    mock_model.clone.assert_called_once()
    cloned_model.set_model_sampler_post_cfg_function.assert_called_once()
    assert result[0] == cloned_model
