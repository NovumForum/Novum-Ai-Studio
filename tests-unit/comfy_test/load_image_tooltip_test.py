import ast
import os


def test_load_image_schema_metadata():
    nodes_py_path = os.path.join(os.path.dirname(__file__), "..", "..", "nodes.py")
    with open(nodes_py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    class_defs = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert "LoadImage" in class_defs
    assert "LoadImageMask" in class_defs

    # 1. LoadImage checks
    load_image_node = class_defs["LoadImage"]
    desc_val = None
    output_tooltips_val = None
    return_types_val = None

    for item in load_image_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if target.id == "DESCRIPTION" and isinstance(item.value, ast.Constant):
                        desc_val = item.value.value
                    elif target.id == "OUTPUT_TOOLTIPS" and isinstance(item.value, ast.Tuple):
                        output_tooltips_val = tuple(elt.value for elt in item.value.elts if isinstance(elt, ast.Constant))
                    elif target.id == "RETURN_TYPES" and isinstance(item.value, ast.Tuple):
                        return_types_val = tuple(elt.value for elt in item.value.elts if isinstance(elt, ast.Constant))

    assert desc_val is not None
    assert "Loads an image" in desc_val
    assert output_tooltips_val is not None
    assert len(output_tooltips_val) == 2
    assert return_types_val == ("IMAGE", "MASK")
    assert len(output_tooltips_val) == len(return_types_val)

    # 2. LoadImageMask checks
    load_image_mask_node = class_defs["LoadImageMask"]
    mask_desc_val = None
    mask_output_tooltips_val = None
    mask_return_types_val = None

    for item in load_image_mask_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if target.id == "DESCRIPTION" and isinstance(item.value, ast.Constant):
                        mask_desc_val = item.value.value
                    elif target.id == "OUTPUT_TOOLTIPS" and isinstance(item.value, ast.Tuple):
                        mask_output_tooltips_val = tuple(elt.value for elt in item.value.elts if isinstance(elt, ast.Constant))
                    elif target.id == "RETURN_TYPES" and isinstance(item.value, ast.Tuple):
                        mask_return_types_val = tuple(elt.value for elt in item.value.elts if isinstance(elt, ast.Constant))

    assert mask_desc_val is not None
    assert "extracts a specific channel" in mask_desc_val
    assert mask_output_tooltips_val is not None
    assert len(mask_output_tooltips_val) == 1
    assert mask_return_types_val == ("MASK",)
    assert len(mask_output_tooltips_val) == len(mask_return_types_val)
