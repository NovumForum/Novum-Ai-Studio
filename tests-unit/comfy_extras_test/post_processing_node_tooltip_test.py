import ast
from pathlib import Path


def get_schema_kwargs(class_node):
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "define_schema":
            for sub_stmt in stmt.body:
                if isinstance(sub_stmt, ast.Return) and isinstance(sub_stmt.value, ast.Call):
                    return {kw.arg: kw.value for kw in sub_stmt.value.keywords}
    return {}


def test_post_processing_nodes_schema_metadata():
    filepath = Path("comfy_extras/nodes_post_processing.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    target_nodes = {
        "Blend": {
            "display_name": "Image Blend",
            "search_aliases_contains": ["blend", "mix images"],
            "inputs": ["image1", "image2", "blend_factor", "blend_mode"],
            "outputs": [("blended", "The resulting blended image tensor.")],
        },
        "Blur": {
            "display_name": "Image Blur",
            "search_aliases_contains": ["blur", "gaussian blur"],
            "inputs": ["image", "blur_radius", "sigma"],
            "outputs": [("blurred", "The blurred output image.")],
        },
        "Quantize": {
            "display_name": "Image Quantize",
            "search_aliases_contains": ["quantize", "color reduction", "dither"],
            "inputs": ["image", "colors", "dither"],
            "outputs": [("quantized", "The color-reduced output image.")],
        },
        "Sharpen": {
            "display_name": "Image Sharpen",
            "search_aliases_contains": ["sharpen", "unsharp mask"],
            "inputs": ["image", "sharpen_radius", "sigma", "alpha"],
            "outputs": [("sharpened", "The sharpened output image.")],
        },
        "ImageScaleToTotalPixels": {
            "display_name": "Image Scale to Megapixels",
            "search_aliases_contains": ["scale to total pixels", "megapixels"],
            "inputs": ["image", "upscale_method", "megapixels", "resolution_steps"],
            "outputs": [("scaled", "The resized output image scaled to target megapixels.")],
        },
    }

    found_nodes = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in target_nodes:
            kwargs = get_schema_kwargs(node)

            assert "display_name" in kwargs, f"{node.name} missing display_name"
            assert isinstance(kwargs["display_name"], ast.Constant)
            assert kwargs["display_name"].value == target_nodes[node.name]["display_name"]

            assert "description" in kwargs, f"{node.name} missing description"
            assert isinstance(kwargs["description"], ast.Constant)
            assert len(kwargs["description"].value) > 10

            assert "search_aliases" in kwargs, f"{node.name} missing search_aliases"
            assert isinstance(kwargs["search_aliases"], ast.List)
            aliases = [el.value for el in kwargs["search_aliases"].elts if isinstance(el, ast.Constant)]
            for expected_alias in target_nodes[node.name]["search_aliases_contains"]:
                assert expected_alias in aliases, f"{expected_alias} missing from search_aliases of {node.name}"

            # Check inputs tooltips
            assert "inputs" in kwargs, f"{node.name} missing inputs"
            assert isinstance(kwargs["inputs"], ast.List)

            input_names_found = []
            for inp_call in kwargs["inputs"].elts:
                if isinstance(inp_call, ast.Call):
                    inp_args = inp_call.args
                    if inp_args and isinstance(inp_args[0], ast.Constant):
                        inp_name = inp_args[0].value
                        input_names_found.append(inp_name)

                    inp_kwargs = {kw.arg: kw.value for kw in inp_call.keywords}
                    assert "tooltip" in inp_kwargs, f"Input '{inp_name}' in {node.name} missing tooltip"
                    assert isinstance(inp_kwargs["tooltip"], ast.Constant)
                    assert len(inp_kwargs["tooltip"].value) > 5

            for expected_inp in target_nodes[node.name]["inputs"]:
                assert expected_inp in input_names_found, f"Input {expected_inp} missing from {node.name}"

            # Check outputs display_name and tooltips
            assert "outputs" in kwargs, f"{node.name} missing outputs"
            assert isinstance(kwargs["outputs"], ast.List)

            for idx, out_call in enumerate(kwargs["outputs"].elts):
                if isinstance(out_call, ast.Call):
                    out_kwargs = {kw.arg: kw.value for kw in out_call.keywords}
                    expected_disp, expected_tip = target_nodes[node.name]["outputs"][idx]

                    assert "display_name" in out_kwargs, f"Output {idx} in {node.name} missing display_name"
                    assert out_kwargs["display_name"].value == expected_disp

                    assert "tooltip" in out_kwargs, f"Output {idx} in {node.name} missing tooltip"
                    assert out_kwargs["tooltip"].value == expected_tip

            found_nodes[node.name] = True

    assert len(found_nodes) == len(target_nodes)
