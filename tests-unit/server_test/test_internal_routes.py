from unittest.mock import MagicMock
import pytest
from api_server.routes.internal.internal_routes import InternalRoutes
import app.logger

pytestmark = pytest.mark.asyncio


@pytest.fixture
def cli(aiohttp_client):
    async def _create_cli():
        prompt_server = MagicMock()
        prompt_server.sockets = {}
        internal_routes = InternalRoutes(prompt_server)
        app_instance = internal_routes.get_app()
        return await aiohttp_client(app_instance)

    return _create_cli


async def test_get_files_invalid_directory_type(cli):
    client = await cli()
    resp = await client.get("/files/invalid_type")
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid directory type"}


async def test_get_files_non_existent_directory(cli, monkeypatch, tmp_path):
    client = await cli()
    non_existent = tmp_path / "does_not_exist"
    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dtype: str(non_existent),
    )
    resp = await client.get("/files/temp")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_get_files_none_directory(cli, monkeypatch):
    client = await cli()
    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dtype: None,
    )
    resp = await client.get("/files/temp")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_get_files_existing_directory(cli, monkeypatch, tmp_path):
    client = await cli()
    valid_dir = tmp_path / "temp_files"
    valid_dir.mkdir()
    (valid_dir / "file1.png").write_text("dummy")
    (valid_dir / ".hidden.png").write_text("hidden")
    (valid_dir / "file2.jpg").write_text("dummy")

    monkeypatch.setattr(
        "api_server.routes.internal.internal_routes.get_directory_by_type",
        lambda dtype: str(valid_dir),
    )
    resp = await client.get("/files/temp")
    assert resp.status == 200
    data = await resp.json()
    assert ".hidden.png" not in data
    assert set(data) == {"file1.png", "file2.jpg"}


async def test_get_logs_when_none(cli, monkeypatch):
    client = await cli()
    monkeypatch.setattr(app.logger, "get_logs", lambda: None)
    resp = await client.get("/logs")
    assert resp.status == 200
    data = await resp.json()
    assert data == ""

    resp_raw = await client.get("/logs/raw")
    assert resp_raw.status == 200
    data_raw = await resp_raw.json()
    assert data_raw["entries"] == []


async def test_get_logs_when_populated(cli, monkeypatch):
    client = await cli()
    fake_logs = [{"t": "2026-11-03T16:00:00", "m": "Log message 1\n"}]
    monkeypatch.setattr(app.logger, "get_logs", lambda: fake_logs)
    resp = await client.get("/logs")
    assert resp.status == 200
    data = await resp.json()
    assert data == "2026-11-03T16:00:00 - Log message 1\n"

    resp_raw = await client.get("/logs/raw")
    assert resp_raw.status == 200
    data_raw = await resp_raw.json()
    assert data_raw["entries"] == fake_logs
