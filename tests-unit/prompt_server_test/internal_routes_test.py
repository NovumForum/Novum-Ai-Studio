import pytest
import os
from unittest.mock import MagicMock, patch
from aiohttp import web
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server

@pytest.fixture
def internal_routes(mock_prompt_server):
    with patch("api_server.routes.internal.internal_routes.TerminalService"):
        ir = InternalRoutes(mock_prompt_server)
        return ir

@pytest.fixture
def app(internal_routes):
    return internal_routes.get_app()

async def test_get_files_valid_directory(aiohttp_client, app, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "image.png").write_text("dummy")
    (output_dir / ".DS_Store").write_text("hidden")

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(output_dir)):
        client = await aiohttp_client(app)
        resp = await client.get("/files/output")
        assert resp.status == 200
        files = await resp.json()
        assert files == ["image.png"]

async def test_get_files_invalid_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}

async def test_get_files_nonexistent_directory(aiohttp_client, app, tmp_path):
    non_existent = tmp_path / "non_existent"
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(non_existent)):
        client = await aiohttp_client(app)
        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found"}
