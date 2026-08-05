from comfy_extras.nodes_logic import SwitchNode, CustomComboNode


class TestLogicNodesTooltip:

    def test_switch_node_schema_ux(self):
        """Verify description, search aliases, and parameter-level tooltips for SwitchNode."""
        schema = SwitchNode.define_schema()

        # Verify node-level metadata
        assert schema.description == "Routes one of two inputs to the output based on a boolean switch value."
        assert "if else" in schema.search_aliases
        assert "conditional" in schema.search_aliases
        assert "toggle" in schema.search_aliases

        # Verify input-level tooltips
        inputs_by_id = {i.id: i for i in schema.inputs}
        assert "switch" in inputs_by_id
        assert inputs_by_id["switch"].tooltip == "When true, 'on_true' is routed to the output; when false, 'on_false' is routed."

        assert "on_false" in inputs_by_id
        assert inputs_by_id["on_false"].tooltip == "The value routed to the output when the switch is false."

        assert "on_true" in inputs_by_id
        assert inputs_by_id["on_true"].tooltip == "The value routed to the output when the switch is true."

        # Verify output-level tooltip
        assert len(schema.outputs) == 1
        output = schema.outputs[0]
        assert output.tooltip == "The selected input value (either 'on_true' or 'on_false')."

    def test_custom_combo_node_schema_ux(self):
        """Verify description, search aliases, and parameter-level tooltips for CustomComboNode."""
        schema = CustomComboNode.define_schema()

        # Verify node-level metadata
        assert schema.description == "Enables users to write and select custom option list values in a dynamic combo dropdown."
        assert "custom dropdown" in schema.search_aliases
        assert "combo select" in schema.search_aliases
        assert "dynamic selection" in schema.search_aliases

        # Verify input-level tooltip
        inputs_by_id = {i.id: i for i in schema.inputs}
        assert "choice" in inputs_by_id
        assert inputs_by_id["choice"].tooltip == "Select the custom option from the user-defined dropdown list."

        # Verify output-level tooltips
        outputs_by_display_name = {o.display_name: o for o in schema.outputs}
        assert "STRING" in outputs_by_display_name
        assert outputs_by_display_name["STRING"].tooltip == "The string value of the selected combo choice."

        assert "INDEX" in outputs_by_display_name
        assert outputs_by_display_name["INDEX"].tooltip == "The zero-based index of the selected combo choice in the options list."
