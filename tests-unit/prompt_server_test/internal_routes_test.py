import pytest
from unittest.mock import patch, MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server

@pytest.fixture
def internal_routes_app(mock_prompt_server):
    ir = InternalRoutes(mock_prompt_server)
    return ir.get_app()

class TestInternalRoutesFiles:
    @pytest.mark.asyncio
    async def test_get_files_valid_directory(self, aiohttp_client, internal_routes_app, tmp_path):
        # Create some temporary files to scan
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "test_file.png").write_text("dummy image data")
        (output_dir / ".hidden_file").write_text("hidden")

        client = await aiohttp_client(internal_routes_app)

        # Mock get_directory_by_type to return our temporary output directory
        with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
            mock_get_dir.side_effect = lambda t: str(output_dir) if t == "output" else None

            resp = await client.get("/files/output")
            assert resp.status == 200
            files = await resp.json()
            assert "test_file.png" in files
            assert ".hidden_file" not in files

    @pytest.mark.asyncio
    async def test_get_files_invalid_directory_type(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)

        resp = await client.get("/files/invalid_type")
        assert resp.status == 400
        data = await resp.json()
        assert data == {"error": "Invalid directory type"}

    @pytest.mark.asyncio
    async def test_get_files_unconfigured_directory(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)

        # Mock get_directory_by_type to return None
        with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
            mock_get_dir.return_value = None

            resp = await client.get("/files/output")
            assert resp.status == 404
            data = await resp.json()
            assert data == {"error": "Directory not found"}

    @pytest.mark.asyncio
    async def test_get_files_non_existent_directory(self, aiohttp_client, internal_routes_app, tmp_path):
        client = await aiohttp_client(internal_routes_app)

        # Mock get_directory_by_type to return a non-existent directory path
        non_existent_path = str(tmp_path / "does_not_exist")
        with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
            mock_get_dir.return_value = non_existent_path

            resp = await client.get("/files/output")
            assert resp.status == 404
            data = await resp.json()
            assert data == {"error": "Directory not found"}
