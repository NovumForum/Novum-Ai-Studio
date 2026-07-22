import pytest
from unittest.mock import MagicMock, patch
from api_server.routes.internal.internal_routes import InternalRoutes

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    server.sockets = {}
    return server

@pytest.fixture
def internal_routes_app(mock_prompt_server):
    routes_handler = InternalRoutes(mock_prompt_server)
    return routes_handler.get_app()

@pytest.mark.asyncio
async def test_get_files_valid_directory(aiohttp_client, internal_routes_app, tmp_path):
    # Setup a temporary directory with some visible and hidden files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.png").write_text("image")
    (input_dir / "file2.jpg").write_text("image")
    (input_dir / ".hidden_file").write_text("hidden")

    # Mock get_directory_by_type to return our temporary directory path
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = str(input_dir)

        client = await aiohttp_client(internal_routes_app)
        resp = await client.get("/files/input")
        assert resp.status == 200
        data = await resp.json()
        assert len(data) == 2
        assert "file1.png" in data
        assert "file2.jpg" in data
        assert ".hidden_file" not in data

@pytest.mark.asyncio
async def test_get_files_none_directory(aiohttp_client, internal_routes_app):
    # Mock get_directory_by_type to return None (unconfigured)
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = None

        client = await aiohttp_client(internal_routes_app)
        resp = await client.get("/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found"}

@pytest.mark.asyncio
async def test_get_files_non_existent_directory(aiohttp_client, internal_routes_app):
    # Mock get_directory_by_type to return a non-existent path
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = "/non/existent/path/comfy_tests"

        client = await aiohttp_client(internal_routes_app)
        resp = await client.get("/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found"}

@pytest.mark.asyncio
async def test_get_files_invalid_directory_type(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}
