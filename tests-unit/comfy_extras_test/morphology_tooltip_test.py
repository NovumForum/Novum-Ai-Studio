import ast
from pathlib import Path


def test_morphology_nodes_schema_metadata_and_tooltips():
    file_path = Path("comfy_extras/nodes_morphology.py")
    assert file_path.exists(), f"File {file_path} does not exist"

    tree = ast.parse(file_path.read_text(), filename=str(file_path))

    expected_nodes = {"Morphology", "ImageRGBToYUV", "ImageYUVToRGB"}
    found_nodes = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in expected_nodes:
            continue

        found_nodes.add(node.name)

        # Find define_schema method
        define_schema_method = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                define_schema_method = item
                break

        assert define_schema_method is not None, (
            f"Node {node.name} missing define_schema method"
        )

        # Find io.Schema call
        schema_call = None
        for stmt in ast.walk(define_schema_method):
            if isinstance(stmt, ast.Call):
                func = stmt.func
                if isinstance(func, ast.Attribute) and func.attr == "Schema":
                    schema_call = stmt
                    break

        assert schema_call is not None, f"Node {node.name} schema call not found"

        keywords = {
            kw.arg: kw.value for kw in schema_call.keywords if kw.arg is not None
        }

        # Check description
        assert "description" in keywords, (
            f"Node {node.name} missing description in schema"
        )
        desc_val = keywords["description"]
        assert (
            isinstance(desc_val, ast.Constant)
            and isinstance(desc_val.value, str)
            and len(desc_val.value) > 10
        )

        # Check search_aliases
        assert "search_aliases" in keywords, (
            f"Node {node.name} missing search_aliases in schema"
        )
        aliases_val = keywords["search_aliases"]
        assert isinstance(aliases_val, ast.List) and len(aliases_val.elts) >= 1

        # Check inputs have tooltip
        assert "inputs" in keywords, f"Node {node.name} missing inputs in schema"
        inputs_list = keywords["inputs"]
        assert isinstance(inputs_list, ast.List)
        for input_call in inputs_list.elts:
            assert isinstance(input_call, ast.Call), (
                f"Input element in {node.name} is not a Call"
            )
            input_kws = {
                kw.arg: kw.value for kw in input_call.keywords if kw.arg is not None
            }
            assert "tooltip" in input_kws, (
                f"Input {input_call} in {node.name} missing tooltip"
            )

        # Check outputs have tooltip
        assert "outputs" in keywords, f"Node {node.name} missing outputs in schema"
        outputs_list = keywords["outputs"]
        assert isinstance(outputs_list, ast.List)
        for output_call in outputs_list.elts:
            assert isinstance(output_call, ast.Call), (
                f"Output element in {node.name} is not a Call"
            )
            output_kws = {
                kw.arg: kw.value for kw in output_call.keywords if kw.arg is not None
            }
            assert "tooltip" in output_kws, (
                f"Output {output_call} in {node.name} missing tooltip"
            )

    assert found_nodes == expected_nodes, (
        f"Expected {expected_nodes}, but found {found_nodes}"
    )
