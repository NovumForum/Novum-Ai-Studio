import pytest
from aiohttp import web
from app.model_manager import ModelFileManager
import folder_paths


@pytest.fixture
def model_file_manager():
    return ModelFileManager()


@pytest.fixture
def app(model_file_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_file_manager.add_routes(routes)
    app.add_routes(routes)
    return app


@pytest.mark.asyncio
async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    # Setup mock folder paths
    mock_dir = tmp_path / "checkpoints"
    mock_dir.mkdir()
    folder_paths.folder_names_and_paths["checkpoints"] = ([str(mock_dir)], set())

    client = await aiohttp_client(app)

    # Attempt path traversal using encoded traversal
    resp = await client.get("/experiment/models/preview/checkpoints/0/..%2f..%2fetc%2fpasswd")
    assert resp.status == 403


@pytest.mark.asyncio
async def test_get_model_preview_invalid_path_index(aiohttp_client, app, tmp_path):
    mock_dir = tmp_path / "checkpoints"
    mock_dir.mkdir()
    folder_paths.folder_names_and_paths["checkpoints"] = ([str(mock_dir)], set())

    client = await aiohttp_client(app)

    # Test out-of-bounds path_index
    resp = await client.get("/experiment/models/preview/checkpoints/999/model.safetensors")
    assert resp.status == 404

    # Test negative path_index
    resp = await client.get("/experiment/models/preview/checkpoints/-1/model.safetensors")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_get_model_preview_invalid_folder_name(aiohttp_client, app):
    client = await aiohttp_client(app)

    resp = await client.get("/experiment/models/preview/invalid_folder_type/0/model.safetensors")
    assert resp.status == 404
