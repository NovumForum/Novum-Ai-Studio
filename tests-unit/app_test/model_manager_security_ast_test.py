import ast
import os
import pytest

def test_model_manager_security_ast():
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "app", "model_manager.py")
    assert os.path.exists(filepath), "model_manager.py must exist"

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    get_model_preview_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_model_preview":
            get_model_preview_node = node
            break

    assert get_model_preview_node is not None, "get_model_preview endpoint must be defined in model_manager.py"

    code_str = ast.unparse(get_model_preview_node)

    # Check for path_index integer conversion try-except handling (returns 400)
    assert "except (ValueError, TypeError):" in code_str or "except Exception:" in code_str or "except (ValueError," in code_str, \
        "get_model_preview must catch integer parsing errors"

    # Check for bounds check
    assert "path_index < 0" in code_str or "path_index >= len(" in code_str, \
        "get_model_preview must validate path_index bounds"

    # Check for path traversal protection using commonpath
    assert "os.path.commonpath" in code_str, \
        "get_model_preview must enforce path traversal protection using os.path.commonpath"
