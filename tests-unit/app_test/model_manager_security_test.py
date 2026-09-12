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
        'checkpoints': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Attempt directory traversal outside tmp_path using URL-encoded slashes to reach handler
        response = await client.get('/experiment/models/preview/checkpoints/0/subfolder%2f..%2f..%2f..%2fetc%2fpasswd')
        assert response.status == 403
        text = await response.text()
        assert "Access denied" in text


async def test_get_model_preview_invalid_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/checkpoints/invalid_index/test.safetensors')
        assert response.status == 400
        text = await response.text()
        assert "Invalid path_index" in text


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/checkpoints/999/test.safetensors')
        assert response.status == 404
