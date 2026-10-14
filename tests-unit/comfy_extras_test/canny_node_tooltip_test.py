import ast
import os


def test_canny_node_schema_metadata():
    filepath = os.path.join(
        os.path.dirname(__file__), "../../comfy_extras/nodes_canny.py"
    )
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    canny_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Canny":
            canny_class = node
            break

    assert canny_class is not None, (
        "Canny class not found in comfy_extras/nodes_canny.py"
    )

    define_schema_fn = None
    for item in canny_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
            define_schema_fn = item
            break

    assert define_schema_fn is not None, "define_schema method not found in Canny class"

    # Find the io.Schema call in return statement
    schema_call = None
    for stmt in define_schema_fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            schema_call = stmt.value
            break

    assert schema_call is not None, "Return io.Schema call not found in define_schema"

    kwargs = {kw.arg: kw.value for kw in schema_call.keywords if kw.arg}

    assert "description" in kwargs, "Canny schema missing 'description'"
    assert isinstance(kwargs["description"], ast.Constant)
    assert len(kwargs["description"].value) > 10

    assert "search_aliases" in kwargs, "Canny schema missing 'search_aliases'"
    assert isinstance(kwargs["search_aliases"], ast.List)
    aliases = [
        elt.value
        for elt in kwargs["search_aliases"].elts
        if isinstance(elt, ast.Constant)
    ]
    assert "edge detection" in aliases

    assert "inputs" in kwargs, "Canny schema missing 'inputs'"
    inputs_list = kwargs["inputs"]
    assert isinstance(inputs_list, ast.List)

    input_tooltips = {}
    for inp in inputs_list.elts:
        if isinstance(inp, ast.Call):
            inp_kwargs = {kw.arg: kw.value for kw in inp.keywords if kw.arg}
            param_name = (
                inp.args[0].value
                if inp.args and isinstance(inp.args[0], ast.Constant)
                else None
            )
            if param_name and "tooltip" in inp_kwargs:
                input_tooltips[param_name] = inp_kwargs["tooltip"].value

    assert "image" in input_tooltips, "Input 'image' missing tooltip"
    assert "low_threshold" in input_tooltips, "Input 'low_threshold' missing tooltip"
    assert "high_threshold" in input_tooltips, "Input 'high_threshold' missing tooltip"

    assert "outputs" in kwargs, "Canny schema missing 'outputs'"
    outputs_list = kwargs["outputs"]
    assert isinstance(outputs_list, ast.List)
    assert len(outputs_list.elts) > 0

    out_call = outputs_list.elts[0]
    assert isinstance(out_call, ast.Call)
    out_kwargs = {kw.arg: kw.value for kw in out_call.keywords if kw.arg}
    assert "tooltip" in out_kwargs, "Output missing tooltip"
    assert len(out_kwargs["tooltip"].value) > 5
