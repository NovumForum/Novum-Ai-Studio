import ast
import os

def test_mask_node_schemas():
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "comfy_extras", "nodes_mask.py")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    target_nodes = {
        "SolidMask": ["value", "width", "height"],
        "InvertMask": ["mask"],
        "CropMask": ["mask", "x", "y", "width", "height"],
        "ThresholdMask": ["mask", "value"],
    }

    found_nodes = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name in target_nodes:
            found_nodes.add(stmt.name)
            expected_inputs = target_nodes[stmt.name]

            # Find define_schema method
            schema_fn = None
            for item in stmt.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    schema_fn = item
                    break

            assert schema_fn is not None, f"{stmt.name} is missing define_schema"

            # Check for return IO.Schema(...)
            return_stmt = None
            for body_item in schema_fn.body:
                if isinstance(body_item, ast.Return):
                    return_stmt = body_item
                    break

            assert return_stmt is not None, f"{stmt.name}.define_schema missing return statement"
            call = return_stmt.value
            assert isinstance(call, ast.Call), f"{stmt.name} return value is not a Call"

            kwargs = {k.arg: k.value for k in call.keywords}

            # Check schema fields
            assert "display_name" in kwargs, f"{stmt.name} missing display_name"
            assert "description" in kwargs, f"{stmt.name} missing description"
            assert "search_aliases" in kwargs, f"{stmt.name} missing search_aliases"
            assert "inputs" in kwargs, f"{stmt.name} missing inputs"
            assert "outputs" in kwargs, f"{stmt.name} missing outputs"

            # Verify inputs have tooltips
            inputs_list = kwargs["inputs"]
            assert isinstance(inputs_list, ast.List), f"{stmt.name} inputs is not a List"

            found_input_names = []
            for input_call in inputs_list.elts:
                assert isinstance(input_call, ast.Call), f"Input element in {stmt.name} is not a Call"
                input_kwargs = {k.arg: k.value for k in input_call.keywords}
                input_args = input_call.args
                assert len(input_args) > 0, f"Input in {stmt.name} missing name argument"
                param_name = ast.literal_eval(input_args[0])
                found_input_names.append(param_name)
                assert "tooltip" in input_kwargs, f"{stmt.name} input '{param_name}' missing tooltip"

            assert set(found_input_names) == set(expected_inputs), f"Mismatch in inputs for {stmt.name}: got {found_input_names}, expected {expected_inputs}"

            # Verify outputs have tooltips
            outputs_list = kwargs["outputs"]
            assert isinstance(outputs_list, ast.List), f"{stmt.name} outputs is not a List"
            for output_call in outputs_list.elts:
                assert isinstance(output_call, ast.Call), f"Output element in {stmt.name} is not a Call"
                output_kwargs = {k.arg: k.value for k in output_call.keywords}
                assert "tooltip" in output_kwargs, f"{stmt.name} output missing tooltip"

    assert found_nodes == set(target_nodes.keys()), f"Missing nodes in test: expected {set(target_nodes.keys())}, found {found_nodes}"
    print("ALL MASK SCHEMA TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_mask_node_schemas()
