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


async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    with patch("folder_paths.folder_names_and_paths", {"test_folder": ([str(tmp_path)], None)}):
        client = await aiohttp_client(app)
        response = await client.get("/experiment/models/preview/test_folder/invalid_idx/model.safetensors")
        assert response.status == 400
        text = await response.text()
        assert "Invalid path_index" in text


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch("folder_paths.folder_names_and_paths", {"test_folder": ([str(tmp_path)], None)}):
        client = await aiohttp_client(app)

        # Index >= length
        response = await client.get("/experiment/models/preview/test_folder/5/model.safetensors")
        assert response.status == 404

        # Negative index
        response_neg = await client.get("/experiment/models/preview/test_folder/-1/model.safetensors")
        assert response_neg.status == 404


async def test_get_model_preview_path_traversal_denied(aiohttp_client, app, tmp_path):
    with patch("folder_paths.folder_names_and_paths", {"test_folder": ([str(tmp_path)], None)}):
        client = await aiohttp_client(app)

        # Attempt path traversal out of tmp_path using encoded slashes/dots or relative subpath
        response = await client.get("/experiment/models/preview/test_folder/0/subfolder%2F..%2F..%2Fetc%2Fpasswd")
        assert response.status == 403
        text = await response.text()
        assert "Access denied" in text
