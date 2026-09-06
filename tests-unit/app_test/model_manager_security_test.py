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


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    sub_folder = tmp_path / "models"
    sub_folder.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("sensitive data")

    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(sub_folder)], None)
    }):
        client = await aiohttp_client(app)
        # Attempt directory traversal out of checkpoints folder
        response = await client.get('/experiment/models/preview/checkpoints/0/../secret.txt')

        # Should return 403 Forbidden due to path traversal restriction
        assert response.status == 403
        assert "Forbidden path traversal" in await response.text()


async def test_get_model_preview_invalid_path_index(aiohttp_client, app):
    client = await aiohttp_client(app)

    # Non-integer path_index
    response = await client.get('/experiment/models/preview/checkpoints/invalid/model.safetensors')
    assert response.status == 400
    assert "Invalid path_index" in await response.text()


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'checkpoints': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Index out of bounds
        response = await client.get('/experiment/models/preview/checkpoints/5/model.safetensors')
        assert response.status == 404

        # Negative out of bounds
        response_neg = await client.get('/experiment/models/preview/checkpoints/-1/model.safetensors')
        assert response_neg.status == 404
