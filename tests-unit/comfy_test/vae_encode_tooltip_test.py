import ast
from pathlib import Path


def test_vae_encode_tooltips_and_description():
    nodes_path = Path(__file__).resolve().parents[2] / "nodes.py"
    with open(nodes_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="nodes.py")

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

    # Check DESCRIPTION
    assert "DESCRIPTION" in class_attrs, "VAEEncode should have a DESCRIPTION attribute"
    assert isinstance(class_attrs["DESCRIPTION"], ast.Constant)
    assert class_attrs["DESCRIPTION"].value == "Encodes pixel space images into latent space representations using a VAE model."

    # Check OUTPUT_TOOLTIPS
    assert "OUTPUT_TOOLTIPS" in class_attrs, "VAEEncode should have an OUTPUT_TOOLTIPS attribute"
    output_tooltips_node = class_attrs["OUTPUT_TOOLTIPS"]
    assert isinstance(output_tooltips_node, ast.Tuple)
    output_tooltips = [elt.value for elt in output_tooltips_node.elts if isinstance(elt, ast.Constant)]
    assert output_tooltips == ["The encoded latent."]

    # Check SEARCH_ALIASES
    assert "SEARCH_ALIASES" in class_attrs, "VAEEncode should have SEARCH_ALIASES"
    aliases_node = class_attrs["SEARCH_ALIASES"]
    assert isinstance(aliases_node, ast.List)
    aliases = [elt.value for elt in aliases_node.elts if isinstance(elt, ast.Constant)]
    assert "encode" in aliases
    assert "encode image" in aliases
    assert "image to latent" in aliases

    # Check INPUT_TYPES tooltips
    assert input_types_fn is not None, "VAEEncode should have INPUT_TYPES method"

    return_dict = None
    for stmt in input_types_fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            return_dict = stmt.value
            break

    assert return_dict is not None, "INPUT_TYPES must return a dict"

    # Find "required" dictionary
    required_dict = None
    for k, v in zip(return_dict.keys, return_dict.values):
        if isinstance(k, ast.Constant) and k.value == "required" and isinstance(v, ast.Dict):
            required_dict = v
            break

    assert required_dict is not None, "'required' input dictionary not found in INPUT_TYPES"

    input_tooltips = {}
    for k, v in zip(required_dict.keys, required_dict.values):
        if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and len(v.elts) > 1:
            cfg = v.elts[1]
            if isinstance(cfg, ast.Dict):
                for ck, cv in zip(cfg.keys, cfg.values):
                    if isinstance(ck, ast.Constant) and ck.value == "tooltip" and isinstance(cv, ast.Constant):
                        input_tooltips[k.value] = cv.value

    assert "pixels" in input_tooltips
    assert input_tooltips["pixels"] == "The image to be encoded into latent space."
    assert "vae" in input_tooltips
    assert input_tooltips["vae"] == "The VAE model used for encoding the image."
