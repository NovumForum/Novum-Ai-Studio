import pytest
import os
from aiohttp import web
from unittest.mock import patch, MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server

@pytest.fixture
def app(mock_prompt_server):
    app = web.Application()
    routes = InternalRoutes(mock_prompt_server)
    routes.setup_routes()
    app.add_routes(routes.routes)
    return app

@pytest.mark.asyncio
async def test_get_files_valid_directory(aiohttp_client, app, tmp_path):
    # Create some dummy files in the tmp_path directory
    file1 = tmp_path / "test_file_1.png"
    file1.write_text("dummy")
    file2 = tmp_path / "test_file_2.png"
    file2.write_text("dummy")
    # A hidden file which should be filtered out
    hidden = tmp_path / ".hidden_file"
    hidden.write_text("hidden")
    # A subfolder which should be filtered out
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(tmp_path)):
        client = await aiohttp_client(app)
        response = await client.get("/files/input")
        assert response.status == 200
        data = await response.json()
        assert "test_file_1.png" in data
        assert "test_file_2.png" in data
        assert ".hidden_file" not in data
        assert "subfolder" not in data

@pytest.mark.asyncio
async def test_get_files_invalid_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get("/files/invalid_type")
    assert response.status == 400
    data = await response.json()
    assert data["error"] == "Invalid directory type"

@pytest.mark.asyncio
async def test_get_files_missing_directory(aiohttp_client, app):
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value="/nonexistent/directory/path"):
        client = await aiohttp_client(app)
        response = await client.get("/files/input")
        assert response.status == 404
        data = await response.json()
        assert data["error"] == "Directory not found"

@pytest.mark.asyncio
async def test_get_files_none_directory(aiohttp_client, app):
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        client = await aiohttp_client(app)
        response = await client.get("/files/input")
        assert response.status == 404
        data = await response.json()
        assert data["error"] == "Directory not found"
