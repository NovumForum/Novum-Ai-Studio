import ast
import os


def test_string_nodes_schema_metadata():
    filepath = os.path.join("comfy_extras", "nodes_string.py")
    assert os.path.exists(filepath), f"{filepath} does not exist"

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    nodes_to_check = [
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

    found_classes = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in nodes_to_check:
            found_classes[node.name] = node

    assert len(found_classes) == len(nodes_to_check), f"Expected {len(nodes_to_check)} string node classes, found {len(found_classes)}"

    for class_name, class_node in found_classes.items():
        # Locate define_schema method
        schema_method = None
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                schema_method = item
                break

        assert schema_method is not None, f"define_schema method not found in {class_name}"

        # Locate io.Schema call inside define_schema
        schema_call = None
        for stmt in schema_method.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "Schema":
                    schema_call = child
                    break
            if schema_call:
                break

        assert schema_call is not None, f"io.Schema call not found in {class_name}"

        # Extract keyword arguments passed to io.Schema
        schema_kwargs = {kw.arg: kw.value for kw in schema_call.keywords if kw.arg}

        # Verify description is present and non-empty
        assert "description" in schema_kwargs, f"{class_name} missing description in io.Schema"
        description_val = schema_kwargs["description"]
        assert isinstance(description_val, ast.Constant) and bool(description_val.value), f"{class_name} description must be a non-empty string"

        # Verify search_aliases is present and non-empty list
        assert "search_aliases" in schema_kwargs, f"{class_name} missing search_aliases in io.Schema"
        aliases_val = schema_kwargs["search_aliases"]
        assert isinstance(aliases_val, ast.List) and len(aliases_val.elts) > 0, f"{class_name} search_aliases must be a non-empty list"

        # Verify input parameter tooltips
        assert "inputs" in schema_kwargs, f"{class_name} missing inputs in io.Schema"
        inputs_val = schema_kwargs["inputs"]
        assert isinstance(inputs_val, ast.List), f"{class_name} inputs must be a list"

        for input_call in inputs_val.elts:
            assert isinstance(input_call, ast.Call), f"Input item in {class_name} must be a Call"
            input_kwargs = {kw.arg: kw.value for kw in input_call.keywords if kw.arg}
            assert "tooltip" in input_kwargs, f"Input param in {class_name} missing tooltip"
            tooltip_val = input_kwargs["tooltip"]
            assert isinstance(tooltip_val, ast.Constant) and bool(tooltip_val.value), f"Input param tooltip in {class_name} must be a non-empty string"
