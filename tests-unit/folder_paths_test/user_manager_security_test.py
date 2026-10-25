import ast
import os

def test_user_manager_ast_security_checks():
    filepath = os.path.join(os.path.dirname(__file__), "../../app/user_manager.py")
    filepath = os.path.abspath(filepath)
    assert os.path.exists(filepath), f"File not found: {filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    # Find listuserdata, get_user_data_path, delete_userdata function nodes
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node

    assert "listuserdata" in functions, "listuserdata function not found"
    assert "get_user_data_path" in functions, "get_user_data_path function not found"
    assert "delete_userdata" in functions, "delete_userdata function not found"

    # Verify listuserdata has a Try handler catching KeyError
    listuserdata_node = functions["listuserdata"]
    try_nodes_list = [n for n in ast.walk(listuserdata_node) if isinstance(n, ast.Try)]
    catches_keyerror_listuserdata = any(
        isinstance(handler.type, ast.Name) and handler.type.id == "KeyError"
        for try_node in try_nodes_list
        for handler in try_node.handlers
    )
    assert catches_keyerror_listuserdata, "listuserdata must catch KeyError"

    # Verify get_user_data_path has a Try handler catching KeyError
    get_user_data_path_node = functions["get_user_data_path"]
    try_nodes_get = [n for n in ast.walk(get_user_data_path_node) if isinstance(n, ast.Try)]
    catches_keyerror_get_path = any(
        isinstance(handler.type, ast.Name) and handler.type.id == "KeyError"
        for try_node in try_nodes_get
        for handler in try_node.handlers
    )
    assert catches_keyerror_get_path, "get_user_data_path must catch KeyError"

    # Verify delete_userdata handles shutil.rmtree and catches FileNotFoundError / OSError
    delete_userdata_node = functions["delete_userdata"]
    try_nodes_del = [n for n in ast.walk(delete_userdata_node) if isinstance(n, ast.Try)]
    assert len(try_nodes_del) > 0, "delete_userdata must contain try-except block"

    caught_exceptions = {
        handler.type.id
        for try_node in try_nodes_del
        for handler in try_node.handlers
        if isinstance(handler.type, ast.Name)
    }
    assert "FileNotFoundError" in caught_exceptions, "delete_userdata must catch FileNotFoundError"
    assert "OSError" in caught_exceptions, "delete_userdata must catch OSError"
