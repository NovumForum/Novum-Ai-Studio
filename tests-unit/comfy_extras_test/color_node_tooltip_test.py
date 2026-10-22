import ast
import os


def test_color_to_rgb_int_schema_tooltips_and_aliases():
    filepath = os.path.join("comfy_extras", "nodes_color.py")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    class_nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ColorToRGBInt"
    ]
    assert len(class_nodes) == 1, "ColorToRGBInt class not found"
    color_node = class_nodes[0]

    schema_call = None
    for subnode in ast.walk(color_node):
        if (
            isinstance(subnode, ast.Call)
            and isinstance(subnode.func, ast.Attribute)
            and subnode.func.attr == "Schema"
        ):
            schema_call = subnode
            break

    assert schema_call is not None, "Schema call not found in ColorToRGBInt"

    kwargs = {kw.arg: kw.value for kw in schema_call.keywords}

    assert "search_aliases" in kwargs, "search_aliases not defined in Schema"
    assert isinstance(kwargs["search_aliases"], ast.List)
    aliases = [elt.value for elt in kwargs["search_aliases"].elts if isinstance(elt, ast.Constant)]
    assert "hex to rgb" in aliases
    assert "hex2rgb" in aliases

    assert "inputs" in kwargs and isinstance(kwargs["inputs"], ast.List)
    input_calls = kwargs["inputs"].elts
    assert len(input_calls) > 0
    color_input = input_calls[0]
    input_kwargs = {kw.arg: kw.value for kw in color_input.keywords}
    assert "tooltip" in input_kwargs, "color input missing tooltip"
    assert isinstance(input_kwargs["tooltip"], ast.Constant)
    assert len(input_kwargs["tooltip"].value) > 0

    assert "outputs" in kwargs and isinstance(kwargs["outputs"], ast.List)
    output_calls = kwargs["outputs"].elts
    assert len(output_calls) > 0
    rgb_output = output_calls[0]
    output_kwargs = {kw.arg: kw.value for kw in rgb_output.keywords}
    assert "tooltip" in output_kwargs, "rgb_int output missing tooltip"
    assert isinstance(output_kwargs["tooltip"], ast.Constant)
    assert len(output_kwargs["tooltip"].value) > 0
