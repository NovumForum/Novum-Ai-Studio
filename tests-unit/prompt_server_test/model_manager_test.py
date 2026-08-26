import os
import sys
from unittest.mock import MagicMock

# Mock comfy.utils and torch before importing app.model_manager
sys.modules['torch'] = MagicMock()
mock_comfy_utils = MagicMock()
mock_comfy_utils.safetensors_header = MagicMock(return_value=None)
sys.modules['comfy.utils'] = mock_comfy_utils

import comfy
comfy.utils = mock_comfy_utils

import pytest
from aiohttp import web
from PIL import Image
import folder_paths
from app.model_manager import ModelFileManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def model_manager(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Setup folder_paths mock/override
    folder_paths_map = {
        "checkpoints": ([str(checkpoints_dir)], {".safetensors", ".ckpt"})
    }
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", folder_paths_map)

    mm = ModelFileManager()
    return mm, checkpoints_dir


@pytest.fixture
def app(model_manager):
    mm, _ = model_manager
    app = web.Application()
    routes = web.RouteTableDef()
    mm.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_get_model_preview_valid(aiohttp_client, app, model_manager):
    _, checkpoints_dir = model_manager
    model_file = checkpoints_dir / "test_model.safetensors"
    preview_file = checkpoints_dir / "test_model.png"

    model_file.write_text("dummy model content")
    img = Image.new("RGB", (10, 10), color="red")
    img.save(preview_file)

    client = await aiohttp_client(app)
    resp = await client.get("/experiment/models/preview/checkpoints/0/test_model.safetensors")
    assert resp.status == 200
    assert resp.content_type == "image/webp"


async def test_get_model_preview_path_traversal(aiohttp_client, app, model_manager):
    client = await aiohttp_client(app)
    # Encoded path traversal attempts
    resp1 = await client.get("/experiment/models/preview/checkpoints/0/..%2F..%2Fetc%2Fpasswd")
    assert resp1.status == 403

    resp2 = await client.get("/experiment/models/preview/checkpoints/0/subfolder%2F..%2F..%2Fetc%2Fpasswd")
    assert resp2.status == 403


async def test_get_model_preview_invalid_path_index(aiohttp_client, app, model_manager):
    client = await aiohttp_client(app)

    resp_out_of_bounds = await client.get("/experiment/models/preview/checkpoints/99/test_model.safetensors")
    assert resp_out_of_bounds.status == 404

    resp_negative = await client.get("/experiment/models/preview/checkpoints/-1/test_model.safetensors")
    assert resp_negative.status == 404

    resp_non_int = await client.get("/experiment/models/preview/checkpoints/invalid/test_model.safetensors")
    assert resp_non_int.status == 404


async def test_get_model_preview_invalid_folder(aiohttp_client, app, model_manager):
    client = await aiohttp_client(app)
    resp = await client.get("/experiment/models/preview/nonexistent_folder/0/test_model.safetensors")
    assert resp.status == 404
