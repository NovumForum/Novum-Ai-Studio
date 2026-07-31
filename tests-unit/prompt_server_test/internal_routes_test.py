import pytest
import os
from unittest.mock import patch, MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio

@pytest.fixture
def app():
    mock_prompt_server = MagicMock()
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()

async def test_get_files_invalid_directory_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}

async def test_get_files_directory_is_none(aiohttp_client, app):
    client = await aiohttp_client(app)
    # Mock get_directory_by_type to return None
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        resp = await client.get("/files/temp")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

async def test_get_files_directory_does_not_exist(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    non_existent_path = str(tmp_path / "does_not_exist")
    # Mock get_directory_by_type to return a path that does not exist
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=non_existent_path):
        resp = await client.get("/files/temp")
        assert resp.status == 404
        data = await resp.json()
        assert "error" in data

async def test_get_files_valid_directory_with_files(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)

    # Create test files
    temp_dir = tmp_path / "temp_dir"
    temp_dir.mkdir()

    file1 = temp_dir / "file1.png"
    file1.touch()

    file2 = temp_dir / "file2.png"
    file2.touch()

    # Make file2 newer than file1
    os.utime(file1, (100, 100))
    os.utime(file2, (200, 200))

    # Create hidden file starting with .
    hidden_file = temp_dir / ".hidden"
    hidden_file.touch()

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(temp_dir)):
        resp = await client.get("/files/temp")
        assert resp.status == 200
        data = await resp.json()
        # Should be sorted newest first (file2.png first, then file1.png), and hidden files excluded
        assert data == ["file2.png", "file1.png"]
