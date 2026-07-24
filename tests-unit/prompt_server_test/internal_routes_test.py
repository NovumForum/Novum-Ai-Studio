import pytest
from api_server.routes.internal.internal_routes import InternalRoutes
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_server():
    server = MagicMock()
    server.sockets = {}
    return server


@pytest.fixture
def internal_app(mock_server):
    internal_routes = InternalRoutes(mock_server)
    return internal_routes.get_app()


async def test_get_files_valid_directory(aiohttp_client, internal_app, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    file1 = output_dir / "test1.png"
    file2 = output_dir / "test2.png"
    hidden_file = output_dir / ".hidden.png"

    file1.write_text("image content 1")
    file2.write_text("image content 2")
    hidden_file.write_text("hidden")

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.side_effect = lambda t: str(output_dir) if t == "output" else None

        client = await aiohttp_client(internal_app)
        resp = await client.get("/files/output")

        assert resp.status == 200
        files = await resp.json()

        assert "test1.png" in files
        assert "test2.png" in files
        assert ".hidden.png" not in files


async def test_get_files_directory_not_configured(aiohttp_client, internal_app):
    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = None

        client = await aiohttp_client(internal_app)
        resp = await client.get("/files/output")

        assert resp.status == 404
        data = await resp.json()
        assert "error" in data


async def test_get_files_directory_does_not_exist(aiohttp_client, internal_app, tmp_path):
    non_existent_dir = tmp_path / "does_not_exist"

    with patch("api_server.routes.internal.internal_routes.get_directory_by_type") as mock_get_dir:
        mock_get_dir.return_value = str(non_existent_dir)

        client = await aiohttp_client(internal_app)
        resp = await client.get("/files/output")

        assert resp.status == 404
        data = await resp.json()
        assert "error" in data


async def test_get_files_invalid_directory_type(aiohttp_client, internal_app):
    client = await aiohttp_client(internal_app)
    resp = await client.get("/files/invalid_type")

    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
