import pytest
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def model_manager():
    return ModelFileManager()


@pytest.fixture
def app(model_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_get_model_preview_path_traversal_forbidden(aiohttp_client, app, tmp_path):
    with patch("folder_paths.folder_names_and_paths", {
        "checkpoints": ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app, auto_decompress=False)
        # %2E%2E%2F escapes client path normalization so it reaches the handler unnormalized
        response = await client.get("/experiment/models/preview/checkpoints/0/%2E%2E%2Fsecret.txt")
        assert response.status == 403


async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get("/experiment/models/preview/checkpoints/invalid/model.safetensors")
    assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch("folder_paths.folder_names_and_paths", {
        "checkpoints": ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get("/experiment/models/preview/checkpoints/5/model.safetensors")
        assert response.status == 404
        response_neg = await client.get("/experiment/models/preview/checkpoints/-1/model.safetensors")
        assert response_neg.status == 404
