import ast
import pytest


def test_feather_mask_ast_structure():
    with open("comfy_extras/nodes_mask.py", "r") as f:
        tree = ast.parse(f.read())

    feather_mask_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FeatherMask":
            feather_mask_cls = node
            break

    assert feather_mask_cls is not None, "FeatherMask class not found in comfy_extras/nodes_mask.py"

    execute_fn = None
    for item in feather_mask_cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "execute":
            execute_fn = item
            break

    assert execute_fn is not None, "execute method not found in FeatherMask class"

    # Verify that the Python 'for' loops over range(left/right/top/bottom) have been removed
    for node in ast.walk(execute_fn):
        if isinstance(node, ast.For):
            # If there's a For loop, check if it iterates over range
            if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', None) == 'range':
                pytest.fail("Found unvectorized Python 'for' loop in FeatherMask.execute")
