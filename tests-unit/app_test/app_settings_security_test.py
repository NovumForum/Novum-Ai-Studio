import ast
import os


def test_app_settings_security_ast():
    file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "app", "app_settings.py"
        )
    )
    assert os.path.exists(file_path), f"File not found: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    post_settings_found = False
    post_setting_found = False

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            if node.name == "post_settings":
                post_settings_found = True
                # Check for try-except block containing request.json()
                has_try_except = any(
                    isinstance(stmt, ast.Try) for stmt in ast.walk(node)
                )
                assert has_try_except, "post_settings must contain a try-except block for request JSON parsing"

                # Check for isinstance check against dict
                has_isinstance_dict = False
                for sub_node in ast.walk(node):
                    if (
                        isinstance(sub_node, ast.Call)
                        and isinstance(sub_node.func, ast.Name)
                        and sub_node.func.id == "isinstance"
                    ):
                        if (
                            len(sub_node.args) >= 2
                            and isinstance(sub_node.args[1], ast.Name)
                            and sub_node.args[1].id == "dict"
                        ):
                            has_isinstance_dict = True
                assert has_isinstance_dict, "post_settings must validate that payload is a dict using isinstance"

            elif node.name == "post_setting":
                post_setting_found = True
                # Check for try-except block containing request.json()
                has_try_except = any(
                    isinstance(stmt, ast.Try) for stmt in ast.walk(node)
                )
                assert has_try_except, "post_setting must contain a try-except block for request JSON parsing"

    assert post_settings_found, "post_settings function not found in AST"
    assert post_setting_found, "post_setting function not found in AST"
