import pytest
from comfy_extras.nodes_logic import SwitchNode, CustomComboNode


def test_switch_node_schema_tooltips_and_aliases():
    """Verify SwitchNode's schema description, search aliases, and input/output tooltips."""
    schema = SwitchNode.define_schema()

    # Assert description and aliases
    assert schema.description == "Select and pass through one of the two inputs based on a boolean condition."
    assert "conditional" in schema.search_aliases
    assert "if else" in schema.search_aliases
    assert "branch" in schema.search_aliases
    assert "toggle" in schema.search_aliases

    # Assert input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "switch" in inputs_dict
    assert inputs_dict["switch"].tooltip == "Condition to control the switch. If True, passes through 'on_true'; if False, passes through 'on_false'."

    assert "on_false" in inputs_dict
    assert inputs_dict["on_false"].tooltip == "The input value to return when the switch is False."

    assert "on_true" in inputs_dict
    assert inputs_dict["on_true"].tooltip == "The input value to return when the switch is True."

    # Assert output tooltip - key by display_name or id
    outputs_dict = {out.display_name: out for out in schema.outputs}
    assert "output" in outputs_dict
    assert outputs_dict["output"].tooltip == "The selected input value (either 'on_true' or 'on_false')."


def test_custom_combo_node_schema_tooltips_and_aliases():
    """Verify CustomComboNode's schema description, search aliases, and input/output tooltips."""
    schema = CustomComboNode.define_schema()

    # Assert description and aliases
    assert schema.description == "Create a custom dropdown menu selection. Allows defining user-specified options directly in the frontend."
    assert "dropdown" in schema.search_aliases
    assert "select list" in schema.search_aliases
    assert "custom option" in schema.search_aliases

    # Assert input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "choice" in inputs_dict
    assert inputs_dict["choice"].tooltip == "The user-specified menu option currently selected."

    # Assert output tooltips - key by display_name or id
    outputs_dict = {out.display_name: out for out in schema.outputs}
    assert "STRING" in outputs_dict
    assert outputs_dict["STRING"].tooltip == "The text of the chosen option."

    assert "INDEX" in outputs_dict
    assert outputs_dict["INDEX"].tooltip == "The zero-based index of the selected option."
