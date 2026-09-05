from unittest.mock import MagicMock
from comfy_extras.nodes_tomesd import TomePatchModel


class TestTomePatchModelNode:

    def test_define_schema_metadata(self):
        """Test TomePatchModel schema metadata including tooltips and search aliases"""
        schema = TomePatchModel.define_schema()

        assert schema.node_id == "TomePatchModel"
        assert schema.display_name == "Apply ToMe (Token Merging)"
        assert schema.category == "model_patches/unet"
        assert "Token Merging (ToMe)" in schema.description
        assert "tome" in schema.search_aliases
        assert "token merging" in schema.search_aliases

        # Verify inputs and tooltips
        input_dict = {inp.id: inp for inp in schema.inputs}
        assert "model" in input_dict
        assert "ratio" in input_dict
        assert input_dict["model"].tooltip == "The UNet diffusion model to patch with Token Merging."
        assert "ratio of tokens to merge" in input_dict["ratio"].tooltip

        # Verify outputs and tooltips
        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The patched model with Token Merging enabled."

    def test_execute_patching(self):
        """Test TomePatchModel.execute clones model and registers attention patches"""
        mock_model = MagicMock()
        cloned_model = MagicMock()
        mock_model.clone.return_value = cloned_model

        result = TomePatchModel.execute(mock_model, ratio=0.3)

        mock_model.clone.assert_called_once()
        cloned_model.set_model_attn1_patch.assert_called_once()
        cloned_model.set_model_attn1_output_patch.assert_called_once()
        assert result.result[0] == cloned_model
