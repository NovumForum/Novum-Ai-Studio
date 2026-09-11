import pytest
import yarl
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

async def test_get_model_preview_path_traversal_rejected(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Construct URL with encoded path traversal to bypass client-side URL normalization
        target_url = yarl.URL('/experiment/models/preview/test_folder/0/..%2f..%2fsecret.png', encoded=True)
        response = await client.get(target_url)
        assert response.status == 403
        assert "Access denied" in await response.text()

async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Non-integer path_index
        response = await client.get('/experiment/models/preview/test_folder/invalid_index/test_model.png')
        assert response.status == 400
        assert "Invalid path_index" in await response.text()

async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Out of bounds path_index
        response = await client.get('/experiment/models/preview/test_folder/999/test_model.png')
        assert response.status == 404
