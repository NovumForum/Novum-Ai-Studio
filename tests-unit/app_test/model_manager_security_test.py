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
    """Verify that path traversal attempts returning files outside the target directory respond with HTTP 403."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        # Construct URL with encoded %2e%2e so aiohttp client does not strip path segments before sending
        url = yarl.URL('/experiment/models/preview/test_folder/0/subfolder/%2e%2e/%2e%2e/etc/passwd', encoded=True)
        response = await client.get(url)
        assert response.status == 403


async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    """Verify that a non-integer path_index responds with HTTP 400 Bad Request."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/invalid_idx/test_model.safetensors')
        assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    """Verify that an out-of-bounds path_index responds with HTTP 404 Not Found."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Test index above bounds
        response1 = await client.get('/experiment/models/preview/test_folder/10/test_model.safetensors')
        assert response1.status == 404

        # Test negative index
        response2 = await client.get('/experiment/models/preview/test_folder/-1/test_model.safetensors')
        assert response2.status == 404
