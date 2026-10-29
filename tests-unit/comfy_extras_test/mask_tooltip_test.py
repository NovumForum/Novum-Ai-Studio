import ast
from pathlib import Path

def get_node_schemas():
    file_path = Path("comfy_extras/nodes_mask.py")
    tree = ast.parse(file_path.read_text())

    schemas = {}

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            class_name = stmt.name
            for item in stmt.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    for child in ast.walk(item):
                        if isinstance(child, ast.Call):
                            # Check if calling IO.Schema
                            func = child.func
                            is_schema_call = False
                            if isinstance(func, ast.Attribute) and func.attr == "Schema":
                                is_schema_call = True

                            if is_schema_call:
                                description = None
                                inputs = []
                                outputs = []

                                for kw in child.keywords:
                                    if kw.arg == "description":
                                        if isinstance(kw.value, ast.Constant):
                                            description = kw.value.value
                                    elif kw.arg == "inputs":
                                        if isinstance(kw.value, ast.List):
                                            for elt in kw.value.elts:
                                                if isinstance(elt, ast.Call):
                                                    # e.g., IO.Latent.Input("destination", tooltip="...")
                                                    param_name = None
                                                    tooltip = None
                                                    if elt.args and isinstance(elt.args[0], ast.Constant):
                                                        param_name = elt.args[0].value
                                                    for input_kw in elt.keywords:
                                                        if input_kw.arg == "tooltip" and isinstance(input_kw.value, ast.Constant):
                                                            tooltip = input_kw.value.value
                                                    if param_name:
                                                        inputs.append((param_name, tooltip))
                                    elif kw.arg == "outputs":
                                        if isinstance(kw.value, ast.List):
                                            for elt in kw.value.elts:
                                                if isinstance(elt, ast.Call):
                                                    tooltip = None
                                                    for out_kw in elt.keywords:
                                                        if out_kw.arg == "tooltip" and isinstance(out_kw.value, ast.Constant):
                                                            tooltip = out_kw.value.value
                                                    outputs.append(tooltip)

                                schemas[class_name] = {
                                    "description": description,
                                    "inputs": inputs,
                                    "outputs": outputs
                                }
    return schemas

def test_mask_node_schemas():
    schemas = get_node_schemas()

    expected_nodes = [
        "LatentCompositeMasked",
        "ImageCompositeMasked",
        "MaskToImage",
        "ImageToMask",
        "ImageColorToMask",
        "SolidMask",
        "InvertMask",
        "CropMask",
        "MaskComposite",
        "FeatherMask",
        "GrowMask",
        "ThresholdMask",
        "MaskPreview",
    ]

    for node in expected_nodes:
        assert node in schemas, f"{node} definition missing from schemas"
        schema = schemas[node]

        assert schema["description"] is not None and len(schema["description"]) > 10, f"{node} missing or short description"

        for param_name, tooltip in schema["inputs"]:
            assert tooltip is not None and len(tooltip) > 5, f"{node} input '{param_name}' missing or short tooltip"

        for output_tooltip in schema["outputs"]:
            assert output_tooltip is not None and len(output_tooltip) > 5, f"{node} output missing or short tooltip"
