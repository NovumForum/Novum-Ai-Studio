import pytest
from aiohttp import web
from unittest.mock import AsyncMock, patch
from app.subgraph_manager import SubgraphManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def subgraph_manager():
    return SubgraphManager()


@pytest.fixture
def app(subgraph_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    subgraph_manager.add_routes(routes, [])
    app.add_routes(routes)
    return app


async def test_get_global_subgraph_not_found(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/global_subgraphs/nonexistent_id")

    assert resp.status == 404
    data = await resp.json()
    assert data == {"error": "Subgraph not found"}


async def test_get_global_subgraph_found(aiohttp_client, app, subgraph_manager):
    sample_entry = {
        "source": "templates",
        "name": "sample",
        "path": "/mock/path.json",
        "info": {"node_pack": "comfyui"},
        "data": '{"test": 123}',
    }
    with patch.object(
        subgraph_manager,
        "get_subgraph",
        new=AsyncMock(return_value=sample_entry),
    ):
        client = await aiohttp_client(app)
        resp = await client.get("/global_subgraphs/valid_id")

        assert resp.status == 200
        data = await resp.json()
        assert data["name"] == "sample"
        assert data["data"] == '{"test": 123}'
        assert "path" not in data  # Path sanitized
