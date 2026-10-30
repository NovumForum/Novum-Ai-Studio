import ast
from pathlib import Path


def _get_node_schema_kwargs(class_node):
    """Finds define_schema method and extracts keyword arguments passed to io.Schema()"""
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
            for stmt in item.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "Schema":
                        kwargs = {kw.arg: kw.value for kw in call.keywords}
                        return call, kwargs
    return None, {}


def test_logic_nodes_schema_metadata():
    filepath = Path("comfy_extras/nodes_logic.py")
    assert filepath.exists(), "nodes_logic.py not found"

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    target_nodes = {
        "SwitchNode": {
            "expected_inputs": ["switch", "on_false", "on_true"],
            "has_search_aliases": True,
        },
        "SoftSwitchNode": {
            "expected_inputs": ["switch", "on_false", "on_true"],
            "has_search_aliases": True,
        },
        "CustomComboNode": {
            "expected_inputs": ["choice"],
            "has_search_aliases": True,
        },
        "ConvertStringToComboNode": {
            "expected_inputs": ["string"],
            "has_search_aliases": True,
        },
        "InvertBooleanNode": {
            "expected_inputs": ["boolean"],
            "has_search_aliases": True,
        },
    }

    found_nodes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in target_nodes:
            call, kwargs = _get_node_schema_kwargs(node)
            found_nodes[node.name] = (call, kwargs)

    assert set(found_nodes.keys()) == set(target_nodes.keys()), f"Missing nodes: {set(target_nodes.keys()) - set(found_nodes.keys())}"

    for node_name, (call, kwargs) in found_nodes.items():
        spec = target_nodes[node_name]

        # 1. Check description
        assert "description" in kwargs, f"{node_name} schema missing 'description'"
        description_val = kwargs["description"]
        assert isinstance(description_val, ast.Constant) and bool(description_val.value), f"{node_name} description must be non-empty string"

        # 2. Check search_aliases
        if spec.get("has_search_aliases"):
            assert "search_aliases" in kwargs, f"{node_name} schema missing 'search_aliases'"
            aliases_val = kwargs["search_aliases"]
            assert isinstance(aliases_val, ast.List) and len(aliases_val.elts) > 0, f"{node_name} search_aliases must be non-empty list"

        # 3. Check inputs tooltips
        assert "inputs" in kwargs, f"{node_name} schema missing 'inputs'"
        inputs_list = kwargs["inputs"]
        assert isinstance(inputs_list, ast.List)

        for input_item in inputs_list.elts:
            assert isinstance(input_item, ast.Call)
            # Find tooltip kwarg in input call
            has_tooltip = any(kw.arg == "tooltip" for kw in input_item.keywords)
            assert has_tooltip, f"{node_name} input call missing 'tooltip' keyword argument"

        # 4. Check outputs tooltips
        assert "outputs" in kwargs, f"{node_name} schema missing 'outputs'"
        outputs_list = kwargs["outputs"]
        assert isinstance(outputs_list, ast.List)

        for output_item in outputs_list.elts:
            assert isinstance(output_item, ast.Call)
            has_tooltip = any(kw.arg == "tooltip" for kw in output_item.keywords)
            assert has_tooltip, f"{node_name} output call missing 'tooltip' keyword argument"
