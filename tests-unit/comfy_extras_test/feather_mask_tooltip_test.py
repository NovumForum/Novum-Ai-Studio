import ast

def test_feather_mask_node_tooltip_schema():
    with open("comfy_extras/nodes_mask.py", "r") as f:
        tree = ast.parse(f.read())

    feather_mask_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatherMask":
            feather_mask_class = node
            break

    assert feather_mask_class is not None, "FeatherMask class not found in comfy_extras/nodes_mask.py"

    define_schema_method = None
    for item in feather_mask_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
            define_schema_method = item
            break

    assert define_schema_method is not None, "define_schema method not found in FeatherMask"

    schema_call = None
    for stmt in define_schema_method.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            schema_call = stmt.value
            break

    assert schema_call is not None, "Schema call return statement not found"

    kwargs = {kw.arg: kw.value for kw in schema_call.keywords}

    assert "display_name" in kwargs, "FeatherMask schema missing display_name"
    assert "description" in kwargs, "FeatherMask schema missing description"
    assert "search_aliases" in kwargs, "FeatherMask schema missing search_aliases"

    inputs_list = kwargs["inputs"]
    assert isinstance(inputs_list, ast.List), "inputs should be a list"

    input_names_with_tooltips = []
    for elt in inputs_list.elts:
        if isinstance(elt, ast.Call):
            input_kwargs = {kw.arg: kw.value for kw in elt.keywords}
            if "tooltip" in input_kwargs:
                param_name = elt.args[0].value if elt.args else None
                input_names_with_tooltips.append(param_name)

    expected_inputs = ["mask", "left", "top", "right", "bottom"]
    for param in expected_inputs:
        assert param in input_names_with_tooltips, f"Input parameter '{param}' is missing a tooltip"

    outputs_list = kwargs["outputs"]
    assert isinstance(outputs_list, ast.List), "outputs should be a list"
    assert len(outputs_list.elts) > 0, "outputs should not be empty"

    out_call = outputs_list.elts[0]
    assert isinstance(out_call, ast.Call), "output should be a Call"
    out_kwargs = {kw.arg: kw.value for kw in out_call.keywords}

    assert "display_name" in out_kwargs, "Output is missing display_name"
    assert "tooltip" in out_kwargs, "Output is missing tooltip"
