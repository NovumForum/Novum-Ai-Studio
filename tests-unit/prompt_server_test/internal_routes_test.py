import pytest
from api_server.routes.internal.internal_routes import InternalRoutes
from unittest.mock import MagicMock

pytestmark = pytest.mark.asyncio


@pytest.fixture
def internal_routes_app():
    # Pass a dummy prompt_server
    prompt_server = MagicMock()
    ir = InternalRoutes(prompt_server)
    app = ir.get_app()
    return app


async def test_get_files_success(
    aiohttp_client, internal_routes_app, tmp_path, monkeypatch
):
    client = await aiohttp_client(internal_routes_app)

    # Create some files in input folder
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.txt").write_text("hello")
    (input_dir / "file2.txt").write_text("world")
    (input_dir / ".hidden").write_text("hidden")

    # Mock get_directory_by_type to return our temp input directory
    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dt: str(input_dir) if dt == "input" else None,
    )

    resp = await client.get("/files/input")
    assert resp.status == 200
    files = await resp.json()
    assert set(files) == {"file1.txt", "file2.txt"}


async def test_get_files_invalid_type(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    resp = await client.get("/files/invalid")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "Invalid directory type"


async def test_get_files_directory_not_configured_or_not_exist(
    aiohttp_client, internal_routes_app, monkeypatch
):
    client = await aiohttp_client(internal_routes_app)

    # Case 1: get_directory_by_type returns None (unconfigured)
    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dt: None,
    )
    resp = await client.get("/files/input")
    assert resp.status == 404
    data = await resp.json()
    assert data["error"] == "Directory not found"

    # Case 2: get_directory_by_type returns a non-existent path
    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dt: "/nonexistent/path/here",
    )
    resp = await client.get("/files/input")
    assert resp.status == 404
    data = await resp.json()
    assert data["error"] == "Directory not found"
