import pytest
import os
from unittest.mock import patch, MagicMock
from aiohttp import web
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server


@pytest.fixture
def app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()


async def test_get_files_invalid_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get('/files/invalid_type')
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"


async def test_get_files_directory_none(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        resp = await client.get('/files/output')
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"


async def test_get_files_directory_not_exists(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    non_existent = str(tmp_path / "does_not_exist")
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=non_existent):
        resp = await client.get('/files/output')
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found"


async def test_get_files_success(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    dir_path = tmp_path / "output_dir"
    dir_path.mkdir()
    file1 = dir_path / "file1.txt"
    file1.write_text("hello")
    file_hidden = dir_path / ".hidden"
    file_hidden.write_text("secret")

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(dir_path)):
        resp = await client.get('/files/output')
        assert resp.status == 200
        data = await resp.json()
        assert "file1.txt" in data
        assert ".hidden" not in data
