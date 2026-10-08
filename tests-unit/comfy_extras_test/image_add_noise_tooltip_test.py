import ast
import os

def test_image_add_noise_schema():
    filepath = os.path.join(os.path.dirname(__file__), "../../comfy_extras/nodes_images.py")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ImageAddNoise":
            found_class = True
            schema_call = None
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if getattr(child.func, "attr", None) == "Schema":
                        schema_call = child
                        break

            assert schema_call is not None, "Schema call not found in ImageAddNoise"

            kwargs = {kw.arg: kw.value for kw in schema_call.keywords}

            assert "display_name" in kwargs, "display_name missing from ImageAddNoise schema"
            assert isinstance(kwargs["display_name"], ast.Constant)
            assert kwargs["display_name"].value == "Add Noise"

            assert "description" in kwargs, "description missing from ImageAddNoise schema"
            assert isinstance(kwargs["description"], ast.Constant)
            assert len(kwargs["description"].value) > 10

            assert "search_aliases" in kwargs, "search_aliases missing from ImageAddNoise schema"
            assert isinstance(kwargs["search_aliases"], ast.List)
            aliases = [elt.value for elt in kwargs["search_aliases"].elts if isinstance(elt, ast.Constant)]
            assert "film grain" in aliases
            assert "add noise" in aliases
            assert "grain" in aliases

            # Verify inputs tooltips
            assert "inputs" in kwargs
            assert isinstance(kwargs["inputs"], ast.List)
            input_tooltips = {}
            for inp in kwargs["inputs"].elts:
                if isinstance(inp, ast.Call):
                    param_name = inp.args[0].value if inp.args else None
                    inp_kwargs = {kw.arg: kw.value for kw in inp.keywords}
                    if "tooltip" in inp_kwargs and isinstance(inp_kwargs["tooltip"], ast.Constant):
                        input_tooltips[param_name] = inp_kwargs["tooltip"].value

            assert "image" in input_tooltips
            assert "strength" in input_tooltips

            # Verify outputs tooltip
            assert "outputs" in kwargs
            assert isinstance(kwargs["outputs"], ast.List)
            output_call = kwargs["outputs"].elts[0]
            out_kwargs = {kw.arg: kw.value for kw in output_call.keywords}
            assert "tooltip" in out_kwargs
            assert isinstance(out_kwargs["tooltip"], ast.Constant)

    assert found_class, "ImageAddNoise class not found"
