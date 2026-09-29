import ast
import os


def test_canny_node_schema_metadata():
    filepath = os.path.join("comfy_extras", "nodes_canny.py")
    assert os.path.exists(filepath), f"{filepath} does not exist"

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    canny_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Canny":
            canny_class = node
            break

    assert canny_class is not None, "Canny class not found in AST"

    define_schema_fn = None
    for item in canny_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
            define_schema_fn = item
            break

    assert define_schema_fn is not None, "define_schema method not found in Canny class"

    # Extract schema keywords and inputs/outputs from return io.Schema(...)
    schema_call = None
    for stmt in define_schema_fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            schema_call = stmt.value
            break

    assert schema_call is not None, "Return io.Schema call not found"

    kwargs = {kw.arg: kw.value for kw in schema_call.keywords}

    # Verify display_name and description
    assert "display_name" in kwargs, "display_name missing in Schema"
    assert isinstance(kwargs["display_name"], ast.Constant)
    assert kwargs["display_name"].value == "Canny Edge Detection"

    assert "description" in kwargs, "description missing in Schema"
    assert isinstance(kwargs["description"], ast.Constant)
    assert "Detects edges" in kwargs["description"].value

    # Verify search_aliases
    assert "search_aliases" in kwargs, "search_aliases missing in Schema"

    # Verify input tooltips
    assert "inputs" in kwargs and isinstance(kwargs["inputs"], ast.List)
    input_tooltips = {}
    for inp in kwargs["inputs"].elts:
        if isinstance(inp, ast.Call):
            name_arg = (
                inp.args[0].value
                if inp.args and isinstance(inp.args[0], ast.Constant)
                else None
            )
            tooltip_kw = next(
                (
                    kw.value.value
                    for kw in inp.keywords
                    if kw.arg == "tooltip" and isinstance(kw.value, ast.Constant)
                ),
                None,
            )
            if name_arg:
                input_tooltips[name_arg] = tooltip_kw

    assert "image" in input_tooltips and input_tooltips["image"] is not None
    assert (
        "low_threshold" in input_tooltips
        and input_tooltips["low_threshold"] is not None
    )
    assert (
        "high_threshold" in input_tooltips
        and input_tooltips["high_threshold"] is not None
    )

    # Verify output tooltips
    assert "outputs" in kwargs and isinstance(kwargs["outputs"], ast.List)
    output_tooltips = []
    for out in kwargs["outputs"].elts:
        if isinstance(out, ast.Call):
            tooltip_kw = next(
                (
                    kw.value.value
                    for kw in out.keywords
                    if kw.arg == "tooltip" and isinstance(kw.value, ast.Constant)
                ),
                None,
            )
            if tooltip_kw:
                output_tooltips.append(tooltip_kw)

    assert len(output_tooltips) > 0, "No output tooltip found"
