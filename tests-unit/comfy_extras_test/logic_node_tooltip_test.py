from comfy_extras.nodes_logic import (
    SwitchNode,
    SoftSwitchNode,
    CustomComboNode,
    ConvertStringToComboNode,
    InvertBooleanNode,
)


def test_logic_nodes_schema_tooltips_and_metadata():
    nodes = [
        SwitchNode,
        SoftSwitchNode,
        CustomComboNode,
        ConvertStringToComboNode,
        InvertBooleanNode,
    ]

    for node_cls in nodes:
        schema = node_cls.define_schema()

        # Check node description
        assert isinstance(schema.description, str) and len(schema.description) > 0, (
            f"Node {node_cls.__name__} missing description"
        )

        # Check search aliases
        assert isinstance(schema.search_aliases, list) and len(schema.search_aliases) > 0, (
            f"Node {node_cls.__name__} missing search_aliases"
        )

        # Check input tooltips
        assert len(schema.inputs) > 0, f"Node {node_cls.__name__} has no inputs"
        for inp in schema.inputs:
            assert isinstance(inp.tooltip, str) and len(inp.tooltip) > 0, (
                f"Node {node_cls.__name__} input '{inp.id}' missing tooltip"
            )

        # Check output tooltips
        assert len(schema.outputs) > 0, f"Node {node_cls.__name__} has no outputs"
        for out in schema.outputs:
            assert isinstance(out.tooltip, str) and len(out.tooltip) > 0, (
                f"Node {node_cls.__name__} output missing tooltip"
            )
