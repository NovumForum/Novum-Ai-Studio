from unittest.mock import MagicMock
from comfy_extras.nodes_freelunch import FreeU, FreeU_V2


def test_freeu_schema():
    schema = FreeU.define_schema()
    assert schema.node_id == "FreeU"
    assert schema.display_name == "FreeU"
    assert "FreeU" in schema.description
    assert "freeu" in schema.search_aliases

    # Check input tooltips
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs_by_id
    assert inputs_by_id["model"].tooltip == "The UNet model to apply FreeU patching to."
    assert inputs_by_id["b1"].tooltip is not None
    assert inputs_by_id["b2"].tooltip is not None
    assert inputs_by_id["s1"].tooltip is not None
    assert inputs_by_id["s2"].tooltip is not None

    # Check output tooltip
    assert len(schema.outputs) == 1
    output = schema.outputs[0]
    assert output.id == "MODEL"
    assert output.display_name == "MODEL"
    assert output.tooltip == "The patched model with FreeU feature adjustments applied."


def test_freeu_v2_schema():
    schema = FreeU_V2.define_schema()
    assert schema.node_id == "FreeU_V2"
    assert schema.display_name == "FreeU V2"
    assert "FreeU V2" in schema.description
    assert "freeu v2" in schema.search_aliases

    # Check input tooltips
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs_by_id
    assert inputs_by_id["model"].tooltip == "The UNet model to apply FreeU V2 patching to."
    assert inputs_by_id["b1"].tooltip is not None
    assert inputs_by_id["b2"].tooltip is not None
    assert inputs_by_id["s1"].tooltip is not None
    assert inputs_by_id["s2"].tooltip is not None

    # Check output tooltip
    assert len(schema.outputs) == 1
    output = schema.outputs[0]
    assert output.id == "MODEL"
    assert output.display_name == "MODEL"
    assert output.tooltip == "The patched model with FreeU V2 feature adjustments applied."


def test_freeu_execution():
    mock_model = MagicMock()
    cloned_model = MagicMock()
    mock_model.clone.return_value = cloned_model

    result = FreeU.execute(mock_model, b1=1.1, b2=1.2, s1=0.9, s2=0.2)
    assert result.args[0] == cloned_model
    cloned_model.set_model_output_block_patch.assert_called_once()


def test_freeu_v2_execution():
    mock_model = MagicMock()
    cloned_model = MagicMock()
    mock_model.clone.return_value = cloned_model

    result = FreeU_V2.execute(mock_model, b1=1.3, b2=1.4, s1=0.9, s2=0.2)
    assert result.args[0] == cloned_model
    cloned_model.set_model_output_block_patch.assert_called_once()
