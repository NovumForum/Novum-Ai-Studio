import os
import pytest
from unittest.mock import patch
from aiohttp import web
import folder_paths
from app.model_manager import ModelFileManager


@pytest.fixture
def app_model_manager(tmp_path):
    manager = ModelFileManager()
    routes = web.RouteTableDef()
    manager.add_routes(routes)

    app = web.Application()
    app.add_routes(routes)

    # Setup dummy model directory
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Create a secret file outside checkpoints directory
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("sensitive_data")

    # Create dummy model file and preview inside checkpoints directory
    model_file = checkpoints_dir / "test_model.safetensors"
    model_file.write_text("model_data")

    folder_paths_dict = {
        "checkpoints": ([str(checkpoints_dir)], {".safetensors"})
    }

    with patch.object(folder_paths, "folder_names_and_paths", folder_paths_dict):
        yield app


@pytest.mark.asyncio
async def test_get_model_preview_path_traversal(aiohttp_client, app_model_manager):
    client = await aiohttp_client(app_model_manager)

    # Attempt path traversal using encoded slash or subpath traversal
    resp = await client.get("/experiment/models/preview/checkpoints/0/subdir%2f%2e%2e%2f%2e%2e%2fsecret.txt")
    assert resp.status == 403


@pytest.mark.asyncio
async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app_model_manager):
    client = await aiohttp_client(app_model_manager)

    resp = await client.get("/experiment/models/preview/checkpoints/invalid/test_model.safetensors")
    assert resp.status == 400


@pytest.mark.asyncio
async def test_get_model_preview_path_index_out_of_bounds(aiohttp_client, app_model_manager):
    client = await aiohttp_client(app_model_manager)

    # Index 5 exceeds available folders array (len = 1)
    resp = await client.get("/experiment/models/preview/checkpoints/5/test_model.safetensors")
    assert resp.status == 404

    # Negative index
    resp_neg = await client.get("/experiment/models/preview/checkpoints/-1/test_model.safetensors")
    assert resp_neg.status == 404
