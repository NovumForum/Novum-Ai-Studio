from comfy_extras.nodes_logic import SwitchNode, CustomComboNode


def test_switch_node_schema_tooltips():
    schema = SwitchNode.GET_SCHEMA()
    assert schema.node_id == "ComfySwitchNode"
    assert schema.description == "Selects and routes between two inputs ('on_true' or 'on_false') based on a boolean switch."
    assert "switch" in schema.search_aliases
    assert "router" in schema.search_aliases

    inputs_dict = {inp.id: inp for inp in schema.inputs}

    assert "switch" in inputs_dict
    assert inputs_dict["switch"].tooltip == "If True, routes 'on_true' to the output. If False, routes 'on_false'."

    assert "on_false" in inputs_dict
    assert inputs_dict["on_false"].tooltip == "The value or stream to output when the switch is False."

    assert "on_true" in inputs_dict
    assert inputs_dict["on_true"].tooltip == "The value or stream to output when the switch is True."


def test_custom_combo_node_schema_tooltips():
    schema = CustomComboNode.GET_SCHEMA()
    assert schema.node_id == "CustomCombo"
    assert schema.description == "A utility node that allows defining and selecting custom options directly from the interface."
    assert "custom combo" in schema.search_aliases

    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "choice" in inputs_dict
    assert inputs_dict["choice"].tooltip == "The selected custom option from the dropdown list."
