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


async def test_get_model_preview_invalid_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/invalid_index/test_model.safetensors')
        assert response.status == 400
        assert "Invalid path_index" in await response.text()


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)

        # Upper bound
        response_high = await client.get('/experiment/models/preview/test_folder/1/test_model.safetensors')
        assert response_high.status == 404

        # Negative bound
        response_neg = await client.get('/experiment/models/preview/test_folder/-1/test_model.safetensors')
        assert response_neg.status == 404


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    # Setup some folders to mock folder names
    base_folder = tmp_path / "models"
    base_folder.mkdir()

    # Target folder inside base_folder
    target_folder = base_folder / "test_target"
    target_folder.mkdir()

    # File outside target_folder but inside tmp_path or parent
    secret_file = tmp_path / "secret.png"
    secret_file.write_text("dummy")

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(target_folder)], None)
    }):
        client = await aiohttp_client(app)

        # Attempt traversal with relative path using URL-encoded '../'
        response_rel = await client.get('/experiment/models/preview/test_folder/0/%2e%2e%2fsecret.png')
        assert response_rel.status == 403
        assert "Access denied" in await response_rel.text()

        # Attempt traversal with absolute path using URL-encoded '/'
        response_abs = await client.get('/experiment/models/preview/test_folder/0/%2fetc%2fpasswd')
        assert response_abs.status == 403
        assert "Access denied" in await response_abs.text()
