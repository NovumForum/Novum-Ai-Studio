import ast
import os


def test_get_model_preview_security_ast():
    model_manager_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "app", "model_manager.py"
    )
    assert os.path.exists(model_manager_path)

    with open(model_manager_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)

    # Find the get_model_preview function definition
    get_model_preview_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_model_preview"
        ):
            get_model_preview_node = node
            break

    assert get_model_preview_node is not None, "get_model_preview endpoint not found"

    # Verify safe integer parsing (try-except block handling ValueError/TypeError)
    has_try_except_val_err = False
    has_status_400 = False
    for node in ast.walk(get_model_preview_node):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                # Check caught exceptions
                if isinstance(handler.type, ast.Tuple):
                    names = [elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)]
                    if "ValueError" in names:
                        has_try_except_val_err = True
                elif isinstance(handler.type, ast.Name) and handler.type.id == "ValueError":
                    has_try_except_val_err = True

                # Check if handler returns HTTP 400
                for body_item in handler.body:
                    if isinstance(body_item, ast.Return) and isinstance(body_item.value, ast.Call):
                        for kw in body_item.value.keywords:
                            if kw.arg == "status" and isinstance(kw.value, ast.Constant) and kw.value.value == 400:
                                has_status_400 = True

    assert has_try_except_val_err, "get_model_preview should safely catch ValueError when parsing path_index"
    assert has_status_400, "get_model_preview should return HTTP 400 when path_index is invalid"

    # Verify bounds check on path_index
    has_bounds_check = False
    for node in ast.walk(get_model_preview_node):
        if isinstance(node, ast.If):
            # Check for path_index comparison
            for subnode in ast.walk(node.test):
                if isinstance(subnode, ast.Name) and subnode.id == "path_index":
                    has_bounds_check = True
                    break

    assert has_bounds_check, "get_model_preview should bounds check path_index"

    # Verify path traversal protection (commonpath check returning 403)
    has_commonpath_check = False
    has_status_403 = False
    for node in ast.walk(get_model_preview_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "commonpath":
                has_commonpath_check = True
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "status" and isinstance(kw.value, ast.Constant) and kw.value.value == 403:
                    has_status_403 = True

    assert has_commonpath_check, "get_model_preview should check os.path.commonpath for path traversal"
    assert has_status_403, "get_model_preview should return HTTP 403 when path traversal is detected"
