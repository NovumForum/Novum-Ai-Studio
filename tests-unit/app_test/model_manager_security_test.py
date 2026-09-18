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


async def test_get_model_preview_invalid_path_index(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get('/experiment/models/preview/test_folder/invalid/test_model.safetensors')
    assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Positive out-of-bounds index
        response = await client.get('/experiment/models/preview/test_folder/999/test_model.safetensors')
        assert response.status == 404

        # Negative index
        response_neg = await client.get('/experiment/models/preview/test_folder/-1/test_model.safetensors')
        assert response_neg.status == 404


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # URL encoded path traversal attempt
        response = await client.get('/experiment/models/preview/test_folder/0/%2e%2e%2fsecret.txt')
        assert response.status == 403
