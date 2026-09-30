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


async def test_get_model_preview_path_traversal_returns_403(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)
        # Attempt path traversal out of model_dir via encoded slash
        response = await client.get('/experiment/models/preview/test_folder/0/..%2fsecret.txt')
        assert response.status == 403

        response2 = await client.get('/experiment/models/preview/test_folder/0/subfolder%2f..%2f..%2fsecret.txt')
        assert response2.status == 403


async def test_get_model_preview_invalid_path_index_type_returns_400(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/not_an_int/test_model.safetensors')
        assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index_returns_404(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Index 99 is out of bounds
        response = await client.get('/experiment/models/preview/test_folder/99/test_model.safetensors')
        assert response.status == 404

        # Negative index out of bounds
        response_neg = await client.get('/experiment/models/preview/test_folder/-1/test_model.safetensors')
        assert response_neg.status == 404
