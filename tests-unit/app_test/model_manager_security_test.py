import pytest
import base64
import json
import struct
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch, MagicMock
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


async def test_invalid_path_index_type_returns_400(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get('/experiment/models/preview/test_folder/not_an_int/model.safetensors')
    assert response.status == 400
    text = await response.text()
    assert "Invalid path_index" in text


async def test_out_of_bounds_path_index_returns_404(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Negative index
        response_neg = await client.get('/experiment/models/preview/test_folder/-1/model.safetensors')
        assert response_neg.status == 404

        # Out of upper bounds index
        response_high = await client.get('/experiment/models/preview/test_folder/999/model.safetensors')
        assert response_high.status == 404


async def test_path_traversal_returns_403(model_manager, tmp_path):
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()

    # Create a dummy request object where filename contains relative path traversal
    request = MagicMock()
    request.match_info = {
        "folder": "test_folder",
        "path_index": "0",
        "filename": "../secret.png"
    }

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(sub_dir)], None)
    }):
        # Mock add_routes or invoke the handler directly
        routes = web.RouteTableDef()
        model_manager.add_routes(routes)
        # Find the route handler registered for preview
        handler = None
        for route in routes:
            if "preview" in route.path:
                handler = route.handler
                break

        assert handler is not None
        response = await handler(request)
        assert response.status == 403
        assert response.text == "Access denied"
