import pytest
import os
from aiohttp import web
from api_server.routes.internal.internal_routes import InternalRoutes
from unittest.mock import patch

@pytest.fixture
def test_app():
    app = web.Application()
    routes = InternalRoutes(prompt_server=None)
    app.add_subapp('/internal', routes.get_app())
    return app

@pytest.mark.asyncio
async def test_get_files_valid(aiohttp_client, test_app, tmp_path):
    # Setup temporary directory with some files
    test_dir = tmp_path / "input"
    test_dir.mkdir()

    # Create some files, including a hidden one
    (test_dir / "file1.txt").write_text("file1 content")
    (test_dir / "file2.jpg").write_text("file2 content")
    (test_dir / ".hidden_file").write_text("hidden")

    # Modify mtimes to enforce sorting: file2.jpg more recent than file1.txt
    os.utime(test_dir / "file1.txt", (1000, 1000))
    os.utime(test_dir / "file2.jpg", (2000, 2000))

    client = await aiohttp_client(test_app)

    # Patch get_directory_by_type to return our test input directory
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.side_effect = lambda t: str(test_dir) if t == "input" else None

        resp = await client.get("/internal/files/input")

        assert resp.status == 200
        files = await resp.json()

        # Hidden file should be filtered out.
        # Sorted by mtime descending, so file2.jpg (2000) comes before file1.txt (1000).
        assert files == ["file2.jpg", "file1.txt"]

@pytest.mark.asyncio
async def test_get_files_invalid_type(aiohttp_client, test_app):
    client = await aiohttp_client(test_app)

    # Try invalid directory type
    resp = await client.get("/internal/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"

@pytest.mark.asyncio
async def test_get_files_unconfigured_or_not_found(aiohttp_client, test_app):
    client = await aiohttp_client(test_app)

    # Try directory type that returns None
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value=None):
        resp = await client.get("/internal/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert "unconfigured or not found" in data["error"]

    # Try directory type that returns a path that is not a directory or does not exist
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type", return_value="/non/existent/path"):
        resp = await client.get("/internal/files/input")
        assert resp.status == 404
        data = await resp.json()
        assert "unconfigured or not found" in data["error"]
