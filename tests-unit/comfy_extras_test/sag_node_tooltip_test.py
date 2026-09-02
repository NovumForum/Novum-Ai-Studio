from unittest.mock import MagicMock
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_sag import SelfAttentionGuidance


def test_self_attention_guidance_schema():
    schema = SelfAttentionGuidance.define_schema()
    assert schema.node_id == "SelfAttentionGuidance"
    assert schema.display_name == "Self-Attention Guidance"
    assert schema.category == "model_patches/unet"
    assert schema.description is not None
    assert "Self-Attention Guidance" in schema.description

    # Search aliases verification
    assert schema.search_aliases is not None
    assert "sag" in schema.search_aliases
    assert "self attention guidance" in schema.search_aliases
    assert "attention guidance" in schema.search_aliases
    assert "structure enhancement" in schema.search_aliases
    assert "detail boost" in schema.search_aliases

    # Inputs verification
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs_dict
    assert inputs_dict["model"].tooltip == "The diffusion model to apply Self-Attention Guidance to."

    assert "scale" in inputs_dict
    assert "Guidance scale factor" in inputs_dict["scale"].tooltip

    assert "blur_sigma" in inputs_dict
    assert "Gaussian blur radius" in inputs_dict["blur_sigma"].tooltip

    # Outputs verification
    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "The patched diffusion model with Self-Attention Guidance enabled."


def test_self_attention_guidance_execute():
    mock_model = MagicMock()
    mock_cloned_model = MagicMock()
    mock_model.clone.return_value = mock_cloned_model

    result = SelfAttentionGuidance.execute(mock_model, scale=0.5, blur_sigma=2.0)

    assert mock_model.clone.called
    assert mock_cloned_model.set_model_sampler_post_cfg_function.called
    assert mock_cloned_model.set_model_attn1_replace.called
    assert result.args[0] == mock_cloned_model
