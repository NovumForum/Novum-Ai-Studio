import sys
import os
import asyncio
import importlib.util

# Ensure repo root is at the beginning of sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Force 'utils' module namespace to point to top-level utils/ package
utils_path = os.path.join(repo_root, "utils")
spec = importlib.util.spec_from_file_location("utils", os.path.join(utils_path, "__init__.py"), submodule_search_locations=[utils_path])
utils_mod = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils_mod
spec.loader.exec_module(utils_mod)

import torch

# Mock torch.cuda for CPU-only environments before importing server
torch.cuda.is_available = lambda: False
torch.cuda.current_device = lambda: 0
torch.cuda.memory_stats = lambda *a, **k: {'reserved_bytes.all.current': 0}
torch.cuda.mem_get_info = lambda *a, **k: (1024**3, 1024**3)
torch.cuda.get_device_properties = lambda *a, **k: type('Props', (), {'total_memory': 1024**3})()

import pytest
from unittest.mock import MagicMock
from aiohttp import web
from server import PromptServer

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_prompt_server():
    loop = asyncio.get_event_loop()
    # Mock PromptQueue and its methods
    mock_queue = MagicMock()
    mock_queue.get_history.return_value = {"prompt1": {"status": "success"}}

    # Instantiate PromptServer with mocked attributes
    server = PromptServer(loop)
    server.prompt_queue = mock_queue
    return server


@pytest.fixture
def history_app(mock_prompt_server):
    app = web.Application()
    routes = web.RouteTableDef()

    @routes.get("/history")
    async def get_history(request):
        max_items = request.rel_url.query.get("max_items", None)
        if max_items is not None:
            try:
                max_items = int(max_items)
                if max_items < 0:
                    return web.json_response({"error": "max_items must be a non-negative integer"}, status=400)
            except (ValueError, TypeError):
                return web.json_response({"error": "max_items must be an integer"}, status=400)

        offset = request.rel_url.query.get("offset", None)
        if offset is not None:
            try:
                offset = int(offset)
            except (ValueError, TypeError):
                return web.json_response({"error": "offset must be an integer"}, status=400)
        else:
            offset = -1

        return web.json_response(mock_prompt_server.prompt_queue.get_history(max_items=max_items, offset=offset))

    app.add_routes(routes)
    return app


async def test_get_history_valid_params(aiohttp_client, history_app, mock_prompt_server):
    client = await aiohttp_client(history_app)
    resp = await client.get("/history?max_items=10&offset=0")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"prompt1": {"status": "success"}}
    mock_prompt_server.prompt_queue.get_history.assert_called_with(max_items=10, offset=0)


async def test_get_history_default_params(aiohttp_client, history_app, mock_prompt_server):
    client = await aiohttp_client(history_app)
    resp = await client.get("/history")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"prompt1": {"status": "success"}}
    mock_prompt_server.prompt_queue.get_history.assert_called_with(max_items=None, offset=-1)


async def test_get_history_invalid_max_items_non_integer(aiohttp_client, history_app):
    client = await aiohttp_client(history_app)
    resp = await client.get("/history?max_items=invalid")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "max_items must be an integer"


async def test_get_history_invalid_max_items_negative(aiohttp_client, history_app):
    client = await aiohttp_client(history_app)
    resp = await client.get("/history?max_items=-5")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "max_items must be a non-negative integer"


async def test_get_history_invalid_offset_non_integer(aiohttp_client, history_app):
    client = await aiohttp_client(history_app)
    resp = await client.get("/history?offset=abc")
    assert resp.status == 400
    data = await resp.json()
    assert data["error"] == "offset must be an integer"
