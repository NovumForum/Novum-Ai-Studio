import pytest
import os
from api_server.routes.internal.internal_routes import InternalRoutes
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.asyncio


@pytest.fixture
def dummy_server():
    return MagicMock()


@pytest.fixture
def internal_app(dummy_server):
    ir = InternalRoutes(dummy_server)
    app = ir.get_app()
    return app


async def test_get_files_invalid_type(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)
    resp = await client.get("/files/not_a_valid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"


async def test_get_files_unconfigured_directory(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)
    # Patch get_directory_by_type to return None
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found or unconfigured"


async def test_get_files_nonexistent_directory(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)
    # Patch get_directory_by_type to return a nonexistent directory
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value="/nonexistent/path/here"):
        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data["error"] == "Directory not found or unconfigured"


async def test_get_files_valid_directory(aiohttp_client, internal_app, tmp_path):
    client = await aiohttp_client(internal_app)

    # Create dummy files
    file1 = tmp_path / "file1.png"
    file2 = tmp_path / "file2.jpg"
    hidden_file = tmp_path / ".hidden_file"

    file1.touch()
    file2.touch()
    hidden_file.touch()

    # Artificially shift modification time of file2 to be more recent than file1
    os.utime(file1, (1000, 1000))
    os.utime(file2, (2000, 2000))

    # Patch get_directory_by_type to return our temporary path
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=str(tmp_path)):
        resp = await client.get("/files/output")
        assert resp.status == 200
        data = await resp.json()
        # Should return visible files sorted by mtime descending (most recent first)
        assert data == ["file2.jpg", "file1.png"]
