import ast
from pathlib import Path


def test_image_compare_schema_metadata():
    node_file = Path("comfy_extras/nodes_image_compare.py")
    assert node_file.exists(), "nodes_image_compare.py should exist"

    tree = ast.parse(node_file.read_text())

    found_node = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ImageCompare":
            found_node = True
            schema_call = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    for stmt in item.body:
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                            schema_call = stmt.value
                            break

            assert schema_call is not None, "ImageCompare should return IO.Schema from define_schema"

            keywords = {kw.arg: kw.value for kw in schema_call.keywords}

            assert "search_aliases" in keywords, "ImageCompare schema should include search_aliases"
            aliases = [elt.value for elt in keywords["search_aliases"].elts]
            assert "compare images" in aliases
            assert "before after" in aliases
            assert "diff images" in aliases
            assert "side by side" in aliases
            assert "slider compare" in aliases

            # Check tooltips on inputs
            inputs_arg = keywords.get("inputs")
            assert inputs_arg is not None and isinstance(inputs_arg, ast.List)

            input_tooltips = {}
            for input_call in inputs_arg.elts:
                if isinstance(input_call, ast.Call):
                    arg_name = input_call.args[0].value if input_call.args else None
                    tooltip = None
                    for kw in input_call.keywords:
                        if kw.arg == "tooltip":
                            tooltip = kw.value.value
                    if arg_name and tooltip:
                        input_tooltips[arg_name] = tooltip

            assert "image_a" in input_tooltips
            assert "image_b" in input_tooltips

    assert found_node, "ImageCompare class should be present in nodes_image_compare.py"


if __name__ == "__main__":
    test_image_compare_schema_metadata()
    print("All image compare tooltip tests passed!")
