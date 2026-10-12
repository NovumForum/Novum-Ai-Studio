import os
import pytest
from unittest.mock import MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes
import folder_paths

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server

@pytest.fixture
def app_internal_routes(mock_prompt_server):
    internal = InternalRoutes(mock_prompt_server)
    return internal.get_app()

class TestInternalRoutesFilesEndpoint:
    @pytest.mark.asyncio
    async def test_get_files_invalid_directory_type(self, aiohttp_client, app_internal_routes):
        client = await aiohttp_client(app_internal_routes)
        resp = await client.get("/files/invalid_type")
        assert resp.status == 400
        data = await resp.json()
        assert data == {"error": "Invalid directory type"}

    @pytest.mark.asyncio
    async def test_get_files_nonexistent_directory(self, aiohttp_client, app_internal_routes, tmp_path, monkeypatch):
        nonexistent_dir = str(tmp_path / "does_not_exist")
        monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: nonexistent_dir)

        client = await aiohttp_client(app_internal_routes)
        resp = await client.get("/files/temp")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_get_files_existing_directory_with_files(self, aiohttp_client, app_internal_routes, tmp_path, monkeypatch):
        test_dir = tmp_path / "temp_files"
        test_dir.mkdir()
        (test_dir / "file1.png").write_text("file1")
        (test_dir / ".hidden_file").write_text("hidden")
        (test_dir / "file2.png").write_text("file2")

        monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(test_dir))

        client = await aiohttp_client(app_internal_routes)
        resp = await client.get("/files/temp")
        assert resp.status == 200
        data = await resp.json()
        assert set(data) == {"file1.png", "file2.png"}
        assert ".hidden_file" not in data
