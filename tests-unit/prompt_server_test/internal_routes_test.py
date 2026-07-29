import pytest
import os
import tempfile
from aiohttp import web
from unittest.mock import MagicMock, patch
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def app():
    # Mock prompt server
    mock_prompt_server = MagicMock()
    internal_routes = InternalRoutes(mock_prompt_server)

    app = web.Application()
    # Build and set up routes
    internal_routes.setup_routes()
    app.add_routes(internal_routes.routes)
    return app


async def test_get_files_valid_directory(aiohttp_client, app, temp_dir):
    # Create test files
    file1 = os.path.join(temp_dir, "file1.txt")
    file2 = os.path.join(temp_dir, "file2.txt")
    hidden_file = os.path.join(temp_dir, ".hidden.txt")

    with open(file1, "w") as f:
        f.write("test1")
    # Make file2 modified later
    with open(file2, "w") as f:
        f.write("test2")
    with open(hidden_file, "w") as f:
        f.write("hidden")

    # Ensure mtimes are different for sorted test
    os.utime(file1, (os.path.getatime(file1), 1000.0))
    os.utime(file2, (os.path.getatime(file2), 2000.0))

    client = await aiohttp_client(app)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = temp_dir

        resp = await client.get("/files/output")
        assert resp.status == 200
        data = await resp.json()

        # Should be sorted by -mtime, meaning file2.txt first, and .hidden.txt excluded
        assert data == ["file2.txt", "file1.txt"]


async def test_get_files_invalid_directory_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}


async def test_get_files_directory_none(aiohttp_client, app):
    client = await aiohttp_client(app)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = None

        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found"}


async def test_get_files_directory_not_exists(aiohttp_client, app):
    client = await aiohttp_client(app)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = "/non/existent/directory/path"

        resp = await client.get("/files/output")
        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found"}
