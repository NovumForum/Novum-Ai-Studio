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


async def test_get_model_preview_path_traversal_returns_403(aiohttp_client, app, tmp_path):
    subfolder = tmp_path / "checkpoints"
    subfolder.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("sensitive data")

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(subfolder)], None)
    }):
        client = await aiohttp_client(app)
        # Attempt path traversal via encoded dotdot in filename parameter
        url = yarl.URL('/experiment/models/preview/test_folder/0/subfolder/%2e%2e/%2e%2e/secret.txt', encoded=True)
        response = await client.get(url)
        assert response.status == 403


async def test_get_model_preview_invalid_path_index_returns_400(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/invalid_idx/model.safetensors')
        assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index_returns_404(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Out-of-bounds positive index
        response = await client.get('/experiment/models/preview/test_folder/999/model.safetensors')
        assert response.status == 404

        # Out-of-bounds negative index
        response = await client.get('/experiment/models/preview/test_folder/-1/model.safetensors')
        assert response.status == 404


async def test_get_model_preview_unknown_folder_returns_404(aiohttp_client, app):
    with patch('folder_paths.folder_names_and_paths', {}):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/nonexistent/0/model.safetensors')
        assert response.status == 404
