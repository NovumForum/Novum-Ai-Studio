import unittest
import ast


class TestInternalRoutesAST(unittest.TestCase):
    def test_internal_routes_ast_validation(self):
        filepath = "api_server/routes/internal/internal_routes.py"
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        # Locate get_files function
        get_files_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_files":
                get_files_node = node
                break

        self.assertIsNotNone(get_files_node, "get_files route handler not found in api_server/routes/internal/internal_routes.py")

        # Check AST for if directory is None or not os.path.exists(directory): return web.json_response([], status=200)
        has_directory_check = False
        for stmt in ast.walk(get_files_node):
            if isinstance(stmt, ast.If):
                code_str = ast.unparse(stmt)
                if "directory is None" in code_str or "not os.path.exists(directory)" in code_str:
                    has_directory_check = True
                    break

        self.assertTrue(has_directory_check, "Missing validation check for directory existence or None in get_files")


if __name__ == "__main__":
    unittest.main()
