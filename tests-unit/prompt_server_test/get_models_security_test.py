import pytest
import sys
import os
import asyncio

# Ensure root directory is in sys.path and pre-import utils to prevent package shadowing during pytest collection
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import utils.install_util  # noqa: F401

from unittest.mock import patch
import torch

# Patch torch CUDA memory & device calls before importing server for CPU-only test environment
patch.object(torch.cuda, "is_available", return_value=False).start()
patch.object(torch.cuda, "current_device", return_value=0).start()
patch.object(torch.cuda, "get_device_properties", return_value=type('props', (), {'total_memory': 8 * 1024**3})()).start()
patch.object(torch.cuda, "memory_stats", return_value={'reserved_bytes.all.current': 0}).start()
patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)).start()

from server import PromptServer
import folder_paths


@pytest.fixture
def prompt_server_app():
    loop = asyncio.get_event_loop_policy().get_event_loop()
    with patch.object(folder_paths, "get_filename_list", return_value=["model1.safetensors"]):
        server = PromptServer(loop)
        server.add_routes()
    return server.app


@pytest.mark.asyncio
async def test_get_models_valid_folder(aiohttp_client, prompt_server_app):
    client = await aiohttp_client(prompt_server_app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        with patch.object(folder_paths, "get_filename_list", return_value=["model1.safetensors", "model2.safetensors"]):
            resp = await client.get("/models/checkpoints")
            assert resp.status == 200
            data = await resp.json()
            assert data == ["model1.safetensors", "model2.safetensors"]


@pytest.mark.asyncio
async def test_get_models_path_traversal_subfolder(aiohttp_client, prompt_server_app):
    client = await aiohttp_client(prompt_server_app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        resp = await client.get("/models/checkpoints/..")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_get_models_invalid_folder(aiohttp_client, prompt_server_app):
    client = await aiohttp_client(prompt_server_app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        resp = await client.get("/models/nonexistent_folder")
        assert resp.status == 404
