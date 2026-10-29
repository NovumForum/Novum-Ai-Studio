import ast


def test_internal_routes_subscribe_logs_ast():
    """Verify via AST that subscribe_logs validates json payload and checks parameter types."""
    with open("api_server/routes/internal/internal_routes.py", "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    subscribe_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "subscribe_logs":
            subscribe_fn = node
            break

    assert subscribe_fn is not None, "subscribe_logs function not found"

    # Verify error handling and validation structure in subscribe_logs
    fn_source = ast.unparse(subscribe_fn)
    assert "Invalid JSON body" in fn_source
    assert "isinstance(json_data, dict)" in fn_source
    assert "clientId" in fn_source
    assert "enabled" in fn_source
    assert "status=400" in fn_source
