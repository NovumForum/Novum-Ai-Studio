import ast
from pathlib import Path


def test_resolution_selector_ast_schema():
    filepath = Path("comfy_extras/nodes_resolution.py")
    tree = ast.parse(filepath.read_text())

    resolution_node_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResolutionSelector":
            resolution_node_found = True
            schema_found = False
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    schema_found = True
                    code_str = ast.unparse(item)
                    assert "search_aliases" in code_str
                    assert "aspect ratio" in code_str
                    assert "megapixels" in code_str
                    assert "dimensions" in code_str
                    assert "resolution" in code_str
            assert schema_found, "define_schema method not found in ResolutionSelector"

    assert resolution_node_found, "ResolutionSelector class not found"
