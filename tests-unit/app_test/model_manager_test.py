import pytest
import base64
import json
import struct
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

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

async def test_get_model_preview_safetensors(aiohttp_client, app, tmp_path):
    img = Image.new('RGB', (100, 100), 'white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    safetensors_file = tmp_path / "test_model.safetensors"
    header_bytes = json.dumps({
        "__metadata__": {
            "ssmd_cover_images": json.dumps([img_b64])
        }
    }).encode('utf-8')
    length_bytes = struct.pack('<Q', len(header_bytes))
    with open(safetensors_file, 'wb') as f:
        f.write(length_bytes)
        f.write(header_bytes)

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/test_model.safetensors')

        # Verify response
        assert response.status == 200
        assert response.content_type == 'image/webp'

        # Verify the response contains valid image data
        img_bytes = BytesIO(await response.read())
        img = Image.open(img_bytes)
        assert img.format
        assert img.format.lower() == 'webp'

        # Clean up
        img.close()


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Test URL-encoded path traversal which gets decoded and routed
        response = await client.get('/experiment/models/preview/test_folder/0/..%2f..%2fetc/passwd')
        assert response.status in (403, 404)

        # Test path traversal with absolute path style or extra slash
        response = await client.get('/experiment/models/preview/test_folder/0//etc/passwd')
        assert response.status in (403, 404)

        # Let's test a direct traversal check by matching with subfolder
        # Since 'filename:.*' is used, any dots that don't simplify out can traverse.
        # But our commonpath check will block them.
        response = await client.get('/experiment/models/preview/test_folder/0/subdir/..%2f..%2fpasswd')
        assert response.status == 403


async def test_get_model_preview_bounds_validation(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Test path_index out of bounds positive
        response = await client.get('/experiment/models/preview/test_folder/1/some_file.png')
        assert response.status == 404

        # Test path_index negative
        response = await client.get('/experiment/models/preview/test_folder/-1/some_file.png')
        assert response.status == 404

        # Test path_index non-integer
        response = await client.get('/experiment/models/preview/test_folder/invalid_index/some_file.png')
        assert response.status == 404
