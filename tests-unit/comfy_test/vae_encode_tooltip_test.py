import ast
import os


def test_vae_encode_schema_and_tooltips():
    nodes_path = os.path.join(os.path.dirname(__file__), "..", "..", "nodes.py")
    with open(nodes_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    vae_encode_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VAEEncode":
            vae_encode_cls = node
            break

    assert vae_encode_cls is not None, "VAEEncode class not found in nodes.py"

    class_attrs = {}
    input_types_fn = None

    for item in vae_encode_cls.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    class_attrs[target.id] = item.value
        elif isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            input_types_fn = item

    # Verify class-level attributes
    assert "DESCRIPTION" in class_attrs, "VAEEncode must define DESCRIPTION"
    assert isinstance(class_attrs["DESCRIPTION"], ast.Constant)
    assert len(class_attrs["DESCRIPTION"].value) > 0

    assert "OUTPUT_TOOLTIPS" in class_attrs, "VAEEncode must define OUTPUT_TOOLTIPS"
    assert isinstance(class_attrs["OUTPUT_TOOLTIPS"], (ast.Tuple, ast.List))
    assert len(class_attrs["OUTPUT_TOOLTIPS"].elts) == 1
    assert isinstance(class_attrs["OUTPUT_TOOLTIPS"].elts[0], ast.Constant)
    assert class_attrs["OUTPUT_TOOLTIPS"].elts[0].value == "The encoded latent."

    assert "SEARCH_ALIASES" in class_attrs, "VAEEncode must define SEARCH_ALIASES"
    assert isinstance(class_attrs["SEARCH_ALIASES"], (ast.Tuple, ast.List))

    # Verify INPUT_TYPES tooltips
    assert input_types_fn is not None, "INPUT_TYPES method not found in VAEEncode"

    return_dict = None
    for stmt in input_types_fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            return_dict = stmt.value
            break

    assert return_dict is not None, "Could not find returned dictionary in INPUT_TYPES"

    required_dict = None
    for key, val in zip(return_dict.keys, return_dict.values):
        if isinstance(key, ast.Constant) and key.value == "required" and isinstance(val, ast.Dict):
            required_dict = val
            break

    assert required_dict is not None, "Could not find 'required' dict in INPUT_TYPES return"

    input_keys = {}
    for key, val in zip(required_dict.keys, required_dict.values):
        if isinstance(key, ast.Constant):
            input_keys[key.value] = val

    assert "pixels" in input_keys, "'pixels' missing from VAEEncode inputs"
    assert "vae" in input_keys, "'vae' missing from VAEEncode inputs"

    # Verify tooltips inside options dictionary for inputs
    for param_name in ["pixels", "vae"]:
        tuple_val = input_keys[param_name]
        assert isinstance(tuple_val, ast.Tuple) and len(tuple_val.elts) >= 2, (
            f"Expected tuple with options for param '{param_name}'"
        )
        opts = tuple_val.elts[1]
        assert isinstance(opts, ast.Dict), f"Expected dict options for param '{param_name}'"

        tooltip_found = False
        for k, v in zip(opts.keys, opts.values):
            if isinstance(k, ast.Constant) and k.value == "tooltip":
                assert isinstance(v, ast.Constant) and len(v.value) > 0
                tooltip_found = True
                break
        assert tooltip_found, f"Param '{param_name}' must have a tooltip"
