import ast
from pathlib import Path


def test_vae_encode_for_inpaint_schema():
    nodes_path = Path(__file__).resolve().parents[2] / "nodes.py"
    with open(nodes_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(nodes_path))

    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VAEEncodeForInpaint":
            class_node = node
            break

    assert class_node is not None, "VAEEncodeForInpaint class not found in nodes.py"

    # Extract class-level attributes
    attributes = {}
    input_types_func = None
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attributes[target.id] = ast.literal_eval(item.value)
        elif isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            input_types_func = item

    assert "DESCRIPTION" in attributes
    assert attributes["DESCRIPTION"] == (
        "Encodes an image and inpaint mask into latent space with optional mask dilation for seamless inpainting."
    )

    assert "SEARCH_ALIASES" in attributes
    assert attributes["SEARCH_ALIASES"] == [
        "inpaint",
        "inpaint encode",
        "mask encode",
        "vae inpaint",
        "image to latent inpaint",
    ]

    assert "RETURN_NAMES" in attributes
    assert attributes["RETURN_NAMES"] == ("latent",)

    assert "OUTPUT_TOOLTIPS" in attributes
    assert attributes["OUTPUT_TOOLTIPS"] == ("The encoded latent containing inpaint noise masking.",)

    # Check INPUT_TYPES returned dictionary
    assert input_types_func is not None, "INPUT_TYPES method not found in VAEEncodeForInpaint"
    ret_stmt = None
    for stmt in input_types_func.body:
        if isinstance(stmt, ast.Return):
            ret_stmt = stmt
            break

    assert ret_stmt is not None, "Return statement not found in INPUT_TYPES"
    input_types_dict = ast.literal_eval(ret_stmt.value)

    assert "required" in input_types_dict
    required = input_types_dict["required"]

    assert "pixels" in required
    assert required["pixels"][0] == "IMAGE"
    assert required["pixels"][1].get("tooltip") == "The image to be encoded into latent space for inpainting."

    assert "vae" in required
    assert required["vae"][0] == "VAE"
    assert required["vae"][1].get("tooltip") == "The VAE model used for encoding the image."

    assert "mask" in required
    assert required["mask"][0] == "MASK"
    assert required["mask"][1].get("tooltip") == "The mask indicating regions to inpaint (white areas will be modified)."

    assert "grow_mask_by" in required
    assert required["grow_mask_by"][0] == "INT"
    assert required["grow_mask_by"][1].get("tooltip") == (
        "Amount of pixels to expand the inpaint mask by to ensure seamless blending in latent space."
    )
