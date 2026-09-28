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

async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Pass encoded path traversal sequence in filename parameter
        response = await client.get('/experiment/models/preview/test_folder/0/..%2F..%2Fsecret.txt')
        assert response.status == 403

async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Pass non-integer path_index
        response = await client.get('/experiment/models/preview/test_folder/invalid_index/test.safetensors')
        assert response.status == 400

async def test_get_model_preview_path_index_out_of_bounds(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Pass out of bounds path_index
        response_neg = await client.get('/experiment/models/preview/test_folder/-1/test.safetensors')
        assert response_neg.status == 404

        response_large = await client.get('/experiment/models/preview/test_folder/999/test.safetensors')
        assert response_large.status == 404
