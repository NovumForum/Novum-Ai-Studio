import pytest
from unittest.mock import patch
from aiohttp import web
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

async def test_get_model_preview_path_traversal_blocked(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("secret_data")

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)

        # Attempt path traversal escaping model_dir
        response = await client.get('/experiment/models/preview/checkpoints/0/subfolder%2F..%2F..%2Fsecret.txt')
        assert response.status == 403

async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)

        response = await client.get('/experiment/models/preview/checkpoints/invalid/model.safetensors')
        assert response.status == 400

async def test_get_model_preview_path_index_out_of_bounds(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)

        response = await client.get('/experiment/models/preview/checkpoints/99/model.safetensors')
        assert response.status == 404

        response_negative = await client.get('/experiment/models/preview/checkpoints/-1/model.safetensors')
        assert response_negative.status == 404
