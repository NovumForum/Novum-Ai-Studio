import pytest
from unittest.mock import MagicMock, patch
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    server.sockets = {}
    return server

@pytest.fixture
def app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()

async def test_get_files_valid_directory(aiohttp_client, app, tmp_path):
    # Mock get_directory_by_type to return our temp path
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(tmp_path)):
        # Create a test file
        test_file = tmp_path / "test.png"
        test_file.touch()

        # Create a hidden file to test is_visible_file filtering
        hidden_file = tmp_path / ".hidden.png"
        hidden_file.touch()

        # Create a subdirectory (should be filtered out by entry.is_file())
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        client = await aiohttp_client(app)
        resp = await client.get("/files/input")
        assert resp.status == 200
        data = await resp.json()
        assert "test.png" in data
        assert ".hidden.png" not in data
        assert "subdir" not in data

async def test_get_files_invalid_directory_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"

async def test_get_files_unconfigured_directory(aiohttp_client, app):
    # Mock get_directory_by_type to return None
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        client = await aiohttp_client(app)
        resp = await client.get("/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"

async def test_get_files_nonexistent_directory(aiohttp_client, app, tmp_path):
    # Mock get_directory_by_type to return a path that does not exist
    nonexistent_path = tmp_path / "does_not_exist"
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(nonexistent_path)):
        client = await aiohttp_client(app)
        resp = await client.get("/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"
