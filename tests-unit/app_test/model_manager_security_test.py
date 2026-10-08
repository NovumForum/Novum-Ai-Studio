import pytest
import os
import base64
import json
import struct
import urllib.parse
from io import BytesIO
from PIL import Image
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


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    target_folder = tmp_path / "models"
    target_folder.mkdir()

    # Create a file outside target_folder that an attacker might try to reach
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret_data")

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(target_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Attempt path traversal via URL-encoded slash in filename parameter
        encoded_filename = urllib.parse.quote("sub/../../secret.txt", safe="")
        resp = await client.get(
            f"/experiment/models/preview/checkpoints/0/{encoded_filename}"
        )
        assert resp.status == 403
        assert await resp.text() == "Access denied"


async def test_get_model_preview_invalid_path_index(aiohttp_client, app, tmp_path):
    target_folder = tmp_path / "models"
    target_folder.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(target_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Invalid path_index (non-integer)
        resp = await client.get(
            "/experiment/models/preview/checkpoints/invalid_index/model.ckpt"
        )
        assert resp.status == 400
        assert await resp.text() == "Invalid path_index"


async def test_get_model_preview_out_of_bounds_path_index(
    aiohttp_client, app, tmp_path
):
    target_folder = tmp_path / "models"
    target_folder.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(target_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Out of bounds path_index
        resp = await client.get(
            "/experiment/models/preview/checkpoints/10/model.ckpt"
        )
        assert resp.status == 404

        # Negative path_index
        resp_neg = await client.get(
            "/experiment/models/preview/checkpoints/-1/model.ckpt"
        )
        assert resp_neg.status == 404


async def test_get_model_preview_valid_request(aiohttp_client, app, tmp_path):
    target_folder = tmp_path / "models"
    target_folder.mkdir()

    img = Image.new("RGB", (50, 50), "blue")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    safetensors_file = target_folder / "test.safetensors"
    header_bytes = json.dumps(
        {"__metadata__": {"ssmd_cover_images": json.dumps([img_b64])}}
    ).encode("utf-8")
    length_bytes = struct.pack("<Q", len(header_bytes))
    with open(safetensors_file, "wb") as f:
        f.write(length_bytes)
        f.write(header_bytes)

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(target_folder)], None)},
    ):
        client = await aiohttp_client(app)

        resp = await client.get(
            "/experiment/models/preview/checkpoints/0/test.safetensors"
        )
        assert resp.status == 200
        assert resp.content_type == "image/webp"
