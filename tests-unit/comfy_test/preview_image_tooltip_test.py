import ast
from pathlib import Path


def test_preview_image_schema_metadata():
    nodes_path = Path(__file__).resolve().parents[2] / "nodes.py"
    assert nodes_path.exists(), f"nodes.py not found at {nodes_path}"

    with open(nodes_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    preview_image_node = None
    for item in tree.body:
        if isinstance(item, ast.ClassDef) and item.name == "PreviewImage":
            preview_image_node = item
            break

    assert preview_image_node is not None, "PreviewImage class not found in nodes.py"

    attributes = {}
    methods = {}
    for item in preview_image_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if isinstance(item.value, ast.Constant):
                        attributes[target.id] = item.value.value
                    elif isinstance(item.value, ast.List):
                        attributes[target.id] = [elt.value for elt in item.value.elts if isinstance(elt, ast.Constant)]
        elif isinstance(item, ast.FunctionDef):
            methods[item.name] = item

    assert "CATEGORY" in attributes, "PreviewImage missing CATEGORY attribute"
    assert attributes["CATEGORY"] == "image"

    assert "ESSENTIALS_CATEGORY" in attributes, "PreviewImage missing ESSENTIALS_CATEGORY attribute"
    assert attributes["ESSENTIALS_CATEGORY"] == "Basics"

    assert "DESCRIPTION" in attributes, "PreviewImage missing DESCRIPTION attribute"
    assert "Displays the input images directly in the ComfyUI user interface" in attributes["DESCRIPTION"]

    assert "SEARCH_ALIASES" in attributes, "PreviewImage missing SEARCH_ALIASES attribute"
    assert "preview" in attributes["SEARCH_ALIASES"]

    assert "INPUT_TYPES" in methods, "PreviewImage missing INPUT_TYPES method"
    input_types_func = methods["INPUT_TYPES"]

    return_dict = None
    for stmt in input_types_func.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            return_dict = stmt.value
            break

    assert return_dict is not None, "INPUT_TYPES does not return a dict"

    # Extract required inputs
    required_dict = None
    for k, v in zip(return_dict.keys, return_dict.values):
        if isinstance(k, ast.Constant) and k.value == "required" and isinstance(v, ast.Dict):
            required_dict = v
            break

    assert required_dict is not None, "'required' key missing in INPUT_TYPES return dict"

    images_spec = None
    for k, v in zip(required_dict.keys, required_dict.values):
        if isinstance(k, ast.Constant) and k.value == "images" and isinstance(v, ast.Tuple):
            images_spec = v
            break

    assert images_spec is not None, "'images' missing from required inputs"
    assert len(images_spec.elts) >= 2, "'images' tuple does not have config dict"
    config_dict = images_spec.elts[1]
    assert isinstance(config_dict, ast.Dict), "'images' second element is not a dict"

    tooltip_found = False
    for k, v in zip(config_dict.keys, config_dict.values):
        if isinstance(k, ast.Constant) and k.value == "tooltip":
            assert isinstance(v, ast.Constant) and isinstance(v.value, str)
            assert "The images to preview in the UI." in v.value
            tooltip_found = True
            break

    assert tooltip_found, "tooltip not found for PreviewImage 'images' input"
