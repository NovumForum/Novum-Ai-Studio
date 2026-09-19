import pytest
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from yarl import URL
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
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.png"

    img = Image.new('RGB', (10, 10), 'red')
    img.save(secret_file, format='PNG')

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)
        # Pass encoded path traversal sequence so client does not strip path segment 0
        url = URL('/experiment/models/preview/checkpoints/0/..%2Fsecret%2Fsecret.png', encoded=True)
        response = await client.get(url)
        assert response.status == 403
        text = await response.text()
        assert "Forbidden" in text

async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/checkpoints/invalid/test.png')
        assert response.status == 400
        text = await response.text()
        assert "Invalid path_index" in text

async def test_get_model_preview_path_index_out_of_bounds(aiohttp_client, app, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(model_dir)], None)
    }):
        client = await aiohttp_client(app)
        response_high = await client.get('/experiment/models/preview/checkpoints/999/test.png')
        assert response_high.status == 404

        response_neg = await client.get('/experiment/models/preview/checkpoints/-1/test.png')
        assert response_neg.status == 404
