import ast
import os


def test_canny_schema_tooltips_and_description():
    file_path = os.path.join("comfy_extras", "nodes_canny.py")
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found_canny = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Canny":
            found_canny = True
            schema_call = None
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and getattr(item.func, "attr", None) == "Schema":
                    schema_call = item
                    break

            assert schema_call is not None, "Canny node must call io.Schema"

            # Verify description keyword arg
            keywords = {kw.arg: kw.value for kw in schema_call.keywords}
            assert "description" in keywords, "Canny schema should have a description"
            assert isinstance(keywords["description"], ast.Constant)
            assert "Canny edge detection" in keywords["description"].value

            # Verify input tooltips
            inputs_arg = keywords.get("inputs")
            assert isinstance(inputs_arg, ast.List)
            assert len(inputs_arg.elts) == 3

            for input_call in inputs_arg.elts:
                assert isinstance(input_call, ast.Call)
                kw_dict = {kw.arg: kw.value for kw in input_call.keywords}
                assert "tooltip" in kw_dict, "Input slot must have a tooltip"
                assert isinstance(kw_dict["tooltip"], ast.Constant)
                assert len(kw_dict["tooltip"].value) > 0

            # Verify output tooltips
            outputs_arg = keywords.get("outputs")
            assert isinstance(outputs_arg, ast.List)
            assert len(outputs_arg.elts) == 1
            output_call = outputs_arg.elts[0]
            assert isinstance(output_call, ast.Call)
            kw_dict = {kw.arg: kw.value for kw in output_call.keywords}
            assert "tooltip" in kw_dict, "Output slot must have a tooltip"
            assert isinstance(kw_dict["tooltip"], ast.Constant)
            assert len(kw_dict["tooltip"].value) > 0

    assert found_canny, "Canny class not found in nodes_canny.py"
