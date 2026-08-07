import pytest
import os
from aiohttp import web
from unittest.mock import patch, MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes


@pytest.fixture
def mock_prompt_server():
    return MagicMock()


@pytest.fixture
def internal_app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()


@pytest.mark.asyncio
async def test_get_files_valid_directory(aiohttp_client, internal_app, tmp_path):
    # Setup a temporary directory to simulate a valid input/output/temp directory
    test_dir = tmp_path / "valid_dir"
    test_dir.mkdir()

    # Create some test files
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")
    # A hidden file that should be filtered out
    (test_dir / ".hidden.txt").write_text("hidden")

    client = await aiohttp_client(internal_app)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = str(test_dir)

        resp = await client.get("/files/input")

        assert resp.status == 200
        data = await resp.json()
        assert "file1.txt" in data
        assert "file2.txt" in data
        assert ".hidden.txt" not in data
        # Check that they are sorted by mtime (most recent first)
        assert len(data) == 2


@pytest.mark.asyncio
async def test_get_files_invalid_type(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)

    resp = await client.get("/files/invalid_type")

    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}


@pytest.mark.asyncio
async def test_get_files_unconfigured_none(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = None

        resp = await client.get("/files/output")

        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found or unconfigured"}


@pytest.mark.asyncio
async def test_get_files_non_existent_directory(aiohttp_client, internal_app, tmp_path):
    client = await aiohttp_client(internal_app)
    non_existent = tmp_path / "non_existent_folder"
    # Ensure it doesn't exist
    if non_existent.exists():
        os.rmdir(non_existent)

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = str(non_existent)

        resp = await client.get("/files/temp")

        assert resp.status == 404
        data = await resp.json()
        assert data == {"error": "Directory not found or unconfigured"}
