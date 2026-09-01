import sys
from unittest.mock import MagicMock

# Mock dependencies before importing comfy modules when running in lightweight test envs
for mod_name in [
    "torch",
    "av",
    "av.container",
    "av.subtitles",
    "av.subtitles.stream",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.PngImagePlugin",
    "tqdm",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from comfy_extras.nodes_logic import SwitchNode, SoftSwitchNode, MISSING


def test_switch_node_schema():
    schema = SwitchNode.define_schema()
    assert schema.node_id == "ComfySwitchNode"
    assert schema.display_name == "Switch"
    assert schema.description == "Conditionally selects between two inputs (on_true or on_false) based on a boolean switch state."
    assert "if" in schema.search_aliases
    assert "conditional" in schema.search_aliases
    assert "branch" in schema.search_aliases

    input_ids = [inp.id for inp in schema.inputs]
    assert "switch" in input_ids
    assert "on_false" in input_ids
    assert "on_true" in input_ids

    switch_inp = next(inp for inp in schema.inputs if inp.id == "switch")
    assert switch_inp.tooltip == "If true, outputs the value connected to on_true; if false, outputs the value connected to on_false."

    on_false_inp = next(inp for inp in schema.inputs if inp.id == "on_false")
    assert on_false_inp.tooltip == "Input value to output when switch is false."

    on_true_inp = next(inp for inp in schema.inputs if inp.id == "on_true")
    assert on_true_inp.tooltip == "Input value to output when switch is true."

    assert len(schema.outputs) == 1
    output = schema.outputs[0]
    assert output.display_name == "output"
    assert output.tooltip == "The selected input value based on the switch condition."


def test_switch_node_execution():
    out_true = SwitchNode.execute(switch=True, on_true="val_true", on_false="val_false")
    assert out_true.args[0] == "val_true"

    out_false = SwitchNode.execute(switch=False, on_true="val_true", on_false="val_false")
    assert out_false.args[0] == "val_false"

    assert SwitchNode.check_lazy_status(switch=True, on_true=None, on_false=None) == ["on_true"]
    assert SwitchNode.check_lazy_status(switch=False, on_true=None, on_false=None) == ["on_false"]


def test_soft_switch_node_schema():
    schema = SoftSwitchNode.define_schema()
    assert schema.node_id == "ComfySoftSwitchNode"
    assert schema.display_name == "Soft Switch"
    assert schema.description == "Conditionally selects between two inputs (on_true or on_false), safely falling back to whichever input is connected if one is missing."
    assert "soft switch" in schema.search_aliases
    assert "fallback switch" in schema.search_aliases

    switch_inp = next(inp for inp in schema.inputs if inp.id == "switch")
    assert switch_inp.tooltip == "If true, outputs on_true; if false, outputs on_false. If only one input is connected, that input is used regardless of switch state."

    on_false_inp = next(inp for inp in schema.inputs if inp.id == "on_false")
    assert on_false_inp.tooltip == "Optional input value to output when switch is false (or as fallback)."

    on_true_inp = next(inp for inp in schema.inputs if inp.id == "on_true")
    assert on_true_inp.tooltip == "Optional input value to output when switch is true (or as fallback)."

    assert len(schema.outputs) == 1
    output = schema.outputs[0]
    assert output.display_name == "output"
    assert output.tooltip == "The selected or fallback input value based on the switch condition."


def test_soft_switch_node_execution():
    # Both inputs connected
    out_true = SoftSwitchNode.execute(switch=True, on_true="val_true", on_false="val_false")
    assert out_true.args[0] == "val_true"

    out_false = SoftSwitchNode.execute(switch=False, on_true="val_true", on_false="val_false")
    assert out_false.args[0] == "val_false"

    # Fallback cases when one input is missing
    out_only_true = SoftSwitchNode.execute(switch=False, on_true="val_true", on_false=MISSING)
    assert out_only_true.args[0] == "val_true"

    out_only_false = SoftSwitchNode.execute(switch=True, on_true=MISSING, on_false="val_false")
    assert out_only_false.args[0] == "val_false"

    # Validation check when both missing
    assert SoftSwitchNode.validate_inputs(switch=True, on_true=MISSING, on_false=MISSING) != True
    assert SoftSwitchNode.validate_inputs(switch=True, on_true="val_true", on_false=MISSING) == True

    # Lazy status checks
    assert SoftSwitchNode.check_lazy_status(switch=True, on_true=MISSING, on_false=None) == ["on_false"]
    assert SoftSwitchNode.check_lazy_status(switch=False, on_true=None, on_false=MISSING) == ["on_true"]
