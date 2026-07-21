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
    routes = InternalRoutes(mock_prompt_server)
    return routes.get_app()

@pytest.mark.asyncio
async def test_get_files_invalid_directory_type(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"

@pytest.mark.asyncio
async def test_get_files_directory_is_none(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"

@pytest.mark.asyncio
async def test_get_files_directory_does_not_exist(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value="/nonexistent/directory/path/here"):
        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"

@pytest.mark.asyncio
async def test_get_files_success(aiohttp_client, internal_routes_app, tmp_path):
    # Create temp files
    file1 = tmp_path / "file1.png"
    file1.touch()
    file2 = tmp_path / "file2.jpg"
    file2.touch()
    # Hidden file should be filtered out
    hidden = tmp_path / ".hidden_file"
    hidden.touch()

    client = await aiohttp_client(internal_routes_app)
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(tmp_path)):
        resp = await client.get("/files/output")
        assert resp.status == 200
        data = await resp.json()
        assert "file1.png" in data
        assert "file2.jpg" in data
        assert ".hidden_file" not in data

@pytest.mark.asyncio
async def test_get_files_os_error(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    # Mocking os.path.isdir to return True so it passes the dir check, but os.scandir to raise OSError
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value="/mock/dir"), \
         patch("os.path.isdir", return_value=True), \
         patch("os.scandir", side_effect=OSError("Permission denied")):
        resp = await client.get("/files/output")
        assert resp.status == 500
        data = await resp.json()
        assert data["error"] == "Failed to read directory"
