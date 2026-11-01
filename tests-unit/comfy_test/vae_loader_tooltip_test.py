import ast
import os


def test_vae_loader_metadata_and_tooltips():
    nodes_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "nodes.py"))
    assert os.path.exists(nodes_py_path), f"{nodes_py_path} does not exist"

    with open(nodes_py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="nodes.py")

    vae_loader_node = None
    for item in tree.body:
        if isinstance(item, ast.ClassDef) and item.name == "VAELoader":
            vae_loader_node = item
            break

    assert vae_loader_node is not None, "VAELoader class definition not found in nodes.py"

    # Verify DESCRIPTION
    desc_node = next(
        (target.value for target in vae_loader_node.body
         if isinstance(target, ast.Assign)
         and any(getattr(t, "id", None) == "DESCRIPTION" for t in target.targets)),
        None
    )
    assert desc_node is not None, "DESCRIPTION attribute missing on VAELoader"
    assert isinstance(desc_node, ast.Constant)
    assert "Variational Autoencoder" in desc_node.value

    # Verify SEARCH_ALIASES
    aliases_node = next(
        (target.value for target in vae_loader_node.body
         if isinstance(target, ast.Assign)
         and any(getattr(t, "id", None) == "SEARCH_ALIASES" for t in target.targets)),
        None
    )
    assert aliases_node is not None, "SEARCH_ALIASES attribute missing on VAELoader"
    assert isinstance(aliases_node, (ast.List, ast.Tuple))
    aliases = [elt.value for elt in aliases_node.elts if isinstance(elt, ast.Constant)]
    assert "load vae" in aliases
    assert "vae loader" in aliases
    assert "vae" in aliases

    # Verify OUTPUT_TOOLTIPS
    tooltips_node = next(
        (target.value for target in vae_loader_node.body
         if isinstance(target, ast.Assign)
         and any(getattr(t, "id", None) == "OUTPUT_TOOLTIPS" for t in target.targets)),
        None
    )
    assert tooltips_node is not None, "OUTPUT_TOOLTIPS attribute missing on VAELoader"
    assert isinstance(tooltips_node, (ast.List, ast.Tuple))
    tooltips = [elt.value for elt in tooltips_node.elts if isinstance(elt, ast.Constant)]
    assert len(tooltips) == 1
    assert "The VAE model" in tooltips[0]

    # Verify INPUT_TYPES tooltip for vae_name
    input_types_fn = next(
        (item for item in vae_loader_node.body
         if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES"),
        None
    )
    assert input_types_fn is not None, "INPUT_TYPES method not found on VAELoader"

    # Look for return dict with required: vae_name
    found_vae_name_tooltip = False
    for subnode in ast.walk(input_types_fn):
        if isinstance(subnode, ast.Dict):
            for k, v in zip(subnode.keys, subnode.values):
                if isinstance(k, ast.Constant) and k.value == "required" and isinstance(v, ast.Dict):
                    for req_k, req_v in zip(v.keys, v.values):
                        if isinstance(req_k, ast.Constant) and req_k.value == "vae_name":
                            # req_v is a Tuple: (s.vae_list(s), {"tooltip": ...})
                            if isinstance(req_v, ast.Tuple) and len(req_v.elts) > 1:
                                opt_dict = req_v.elts[1]
                                if isinstance(opt_dict, ast.Dict):
                                    for opt_k, opt_val in zip(opt_dict.keys, opt_dict.values):
                                        if isinstance(opt_k, ast.Constant) and opt_k.value == "tooltip":
                                            if isinstance(opt_val, ast.Constant) and "The name of the VAE model to load." in opt_val.value:
                                                found_vae_name_tooltip = True

    assert found_vae_name_tooltip, "vae_name tooltip not found or incorrect in VAELoader.INPUT_TYPES"
