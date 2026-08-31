import pytest
import os
from aiohttp import web
from app.model_manager import ModelFileManager
import folder_paths
from unittest.mock import patch

pytestmark = pytest.mark.asyncio


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


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    with patch.dict(folder_paths.folder_names_and_paths, {"checkpoints": ([str(checkpoints_dir)], {".safetensors"})}):
        client = await aiohttp_client(app)
        # Attempt path traversal out of checkpoints directory using URL-encoded dots
        resp = await client.get("/experiment/models/preview/checkpoints/0/%2e%2e%2fsecret.txt")
        assert resp.status == 403


async def test_get_model_preview_path_index_out_of_bounds(aiohttp_client, app, tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    with patch.dict(folder_paths.folder_names_and_paths, {"checkpoints": ([str(checkpoints_dir)], {".safetensors"})}):
        client = await aiohttp_client(app)
        # Invalid negative path_index
        resp = await client.get("/experiment/models/preview/checkpoints/-1/model.safetensors")
        assert resp.status == 404

        # Invalid out of range path_index
        resp = await client.get("/experiment/models/preview/checkpoints/5/model.safetensors")
        assert resp.status == 404


async def test_get_model_preview_invalid_folder_name(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/experiment/models/preview/invalid_folder/0/model.safetensors")
    assert resp.status == 404
