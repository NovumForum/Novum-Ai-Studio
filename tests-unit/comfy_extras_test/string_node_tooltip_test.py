import ast
import os


def test_string_nodes_metadata_and_tooltips():
    filepath = os.path.join("comfy_extras", "nodes_string.py")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    node_classes = [
        "StringConcatenate",
        "StringSubstring",
        "StringLength",
        "CaseConverter",
        "StringTrim",
        "StringReplace",
        "StringContains",
        "StringCompare",
        "RegexMatch",
        "RegexExtract",
        "RegexReplace",
    ]

    found_classes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in node_classes:
            found_classes.add(node.name)

            # Find define_schema method
            schema_method = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    schema_method = item
                    break

            assert schema_method is not None, f"{node.name} missing define_schema"

            # Check io.Schema call arguments
            schema_call = None
            for stmt in ast.walk(schema_method):
                if isinstance(stmt, ast.Call) and getattr(stmt.func, "attr", None) == "Schema":
                    schema_call = stmt
                    break

            assert schema_call is not None, f"{node.name} missing io.Schema call"

            kwargs = {kw.arg: kw.value for kw in schema_call.keywords if kw.arg}

            # Verify description and search_aliases
            assert "description" in kwargs, f"{node.name} missing description in Schema"
            assert isinstance(kwargs["description"], ast.Constant) and isinstance(kwargs["description"].value, str), f"{node.name} description must be string"
            assert len(kwargs["description"].value.strip()) > 0, f"{node.name} description empty"

            assert "search_aliases" in kwargs, f"{node.name} missing search_aliases in Schema"

            # Verify input parameter tooltips
            assert "inputs" in kwargs, f"{node.name} missing inputs in Schema"
            inputs_list = kwargs["inputs"]
            assert isinstance(inputs_list, ast.List), f"{node.name} inputs must be a list"

            for input_call in inputs_list.elts:
                assert isinstance(input_call, ast.Call), f"Input element in {node.name} is not a Call"
                input_kwargs = {kw.arg: kw.value for kw in input_call.keywords if kw.arg}
                assert "tooltip" in input_kwargs, f"Input in {node.name} missing tooltip keyword arg"
                tooltip_val = input_kwargs["tooltip"]
                assert isinstance(tooltip_val, ast.Constant) and isinstance(tooltip_val.value, str), f"Tooltip in {node.name} must be a string constant"
                assert len(tooltip_val.value.strip()) > 0, f"Tooltip in {node.name} is empty"

    assert found_classes == set(node_classes), f"Missing node classes: {set(node_classes) - found_classes}"
