from comfy_extras.nodes_logic import SwitchNode, CustomComboNode


class TestLogicNodesTooltip:

    def test_switch_node_schema_metadata(self):
        """Verify SwitchNode has accurate description, search aliases, and parameter-level tooltips."""
        schema = SwitchNode.define_schema()

        # Check node description & aliases
        assert schema.description == "Route and evaluate one of two inputs depending on a boolean switch value."
        assert set(schema.search_aliases) == {"if else", "conditional", "route", "select input", "toggle route", "branch"}

        # Check input parameter tooltips
        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert "switch" in inputs_dict
        assert inputs_dict["switch"].tooltip == "When enabled (True), the 'on_true' input is evaluated and returned. When disabled (False), the 'on_false' input is evaluated and returned."

        assert "on_false" in inputs_dict
        assert inputs_dict["on_false"].tooltip == "The input value evaluated and returned when the switch is disabled (False)."

        assert "on_true" in inputs_dict
        assert inputs_dict["on_true"].tooltip == "The input value evaluated and returned when the switch is enabled (True)."

        # Check output parameter tooltips
        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "The evaluated output value matching the active branch (either 'on_true' or 'on_false')."

    def test_custom_combo_node_schema_metadata(self):
        """Verify CustomComboNode has accurate description, search aliases, and parameter-level tooltips."""
        schema = CustomComboNode.define_schema()

        # Check node description & aliases
        assert schema.description == "An experimental node that allows user-defined custom dropdown menu options in the frontend interface."
        assert set(schema.search_aliases) == {"custom dropdown", "user options", "custom selection", "manual combo"}

        # Check input parameter tooltip
        inputs_dict = {inp.id: inp for inp in schema.inputs}
        assert "choice" in inputs_dict
        assert inputs_dict["choice"].tooltip == "The selected custom option value. Options can be customized directly in the frontend."

        # Check output parameter tooltips
        outputs_dict = {out.id: out for out in schema.outputs}
        assert "STRING" in outputs_dict
        assert outputs_dict["STRING"].tooltip == "The string value of the currently selected dropdown option."

        assert "INDEX" in outputs_dict
        assert outputs_dict["INDEX"].tooltip == "The 0-based index of the currently selected dropdown option."
