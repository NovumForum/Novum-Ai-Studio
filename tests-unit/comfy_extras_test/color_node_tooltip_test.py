import ast
from pathlib import Path


def test_color_to_rgb_int_node_schema():
    filepath = Path("comfy_extras/nodes_color.py")
    assert filepath.exists(), f"File {filepath} does not exist"

    tree = ast.parse(filepath.read_text())

    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    color_class = next((node for node in class_nodes if node.name == "ColorToRGBInt"), None)
    assert color_class is not None, "ColorToRGBInt class not found"

    # Find define_schema method
    define_schema_func = next(
        (
            item
            for item in color_class.body
            if isinstance(item, ast.FunctionDef) and item.name == "define_schema"
        ),
        None,
    )
    assert define_schema_func is not None, "define_schema method not found"

    # Inspect return io.Schema(...) call keywords
    return_stmt = next(
        (node for node in define_schema_func.body if isinstance(node, ast.Return)),
        None,
    )
    assert return_stmt is not None, "Return statement not found in define_schema"
    assert isinstance(return_stmt.value, ast.Call)

    keywords = {keyword.arg: keyword.value for keyword in return_stmt.value.keywords}

    # Check search_aliases
    assert "search_aliases" in keywords, "search_aliases keyword missing in io.Schema"
    aliases = [elt.value for elt in keywords["search_aliases"].elts]
    assert "hex to int" in aliases
    assert "color picker" in aliases
    assert "color to integer" in aliases

    # Check inputs tooltips
    inputs_list = keywords["inputs"]
    assert isinstance(inputs_list, ast.List)
    assert len(inputs_list.elts) > 0
    input_call = inputs_list.elts[0]
    input_kw = {keyword.arg: keyword.value.value for keyword in input_call.keywords if isinstance(keyword.value, ast.Constant)}
    assert "tooltip" in input_kw, "Input 'color' tooltip missing"
    assert "Hex color code" in input_kw["tooltip"]

    # Check outputs tooltips
    outputs_list = keywords["outputs"]
    assert isinstance(outputs_list, ast.List)
    assert len(outputs_list.elts) > 0
    output_call = outputs_list.elts[0]
    output_kw = {keyword.arg: keyword.value.value for keyword in output_call.keywords if isinstance(keyword.value, ast.Constant)}
    assert "tooltip" in output_kw, "Output 'rgb_int' tooltip missing"
    assert "Integer representation" in output_kw["tooltip"]
