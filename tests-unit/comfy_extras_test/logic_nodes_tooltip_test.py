from comfy_extras.nodes_logic import SwitchNode, CustomComboNode

class TestLogicNodesTooltip:
    def test_switch_node_schema(self):
        """Verify SwitchNode's schema description, search aliases, and input parameter tooltips."""
        schema = SwitchNode.define_schema()

        # Verify node metadata
        assert schema.node_id == "ComfySwitchNode"
        assert schema.display_name == "Switch"
        assert schema.description == "Route and evaluate one of two inputs depending on a boolean switch value."
        assert "if else" in schema.search_aliases
        assert "conditional" in schema.search_aliases
        assert "route" in schema.search_aliases

        # Find inputs
        inputs = {inp.id: inp for inp in schema.inputs}
        assert "switch" in inputs
        assert "on_false" in inputs
        assert "on_true" in inputs

        # Verify tooltips
        assert "on_true" in inputs["switch"].tooltip
        assert "disabled" in inputs["on_false"].tooltip
        assert "enabled" in inputs["on_true"].tooltip

    def test_custom_combo_node_schema(self):
        """Verify CustomComboNode's schema description, search aliases, and input parameter tooltips."""
        schema = CustomComboNode.define_schema()

        # Verify node metadata
        assert schema.node_id == "CustomCombo"
        assert schema.display_name == "Custom Combo"
        assert schema.description == "An experimental node that allows user-defined custom dropdown menu options in the frontend interface."
        assert "custom dropdown" in schema.search_aliases
        assert "user options" in schema.search_aliases

        # Find inputs
        inputs = {inp.id: inp for inp in schema.inputs}
        assert "choice" in inputs

        # Verify tooltips
        assert "customized directly" in inputs["choice"].tooltip
