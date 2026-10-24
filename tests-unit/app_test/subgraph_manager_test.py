import pytest
import json
from unittest.mock import MagicMock
from aiohttp import web
from app.subgraph_manager import SubgraphManager


@pytest.fixture
def subgraph_manager():
    return SubgraphManager()


@pytest.mark.asyncio
async def test_get_global_subgraph_found(subgraph_manager):
    routes = web.RouteTableDef()
    loaded_modules = {}
    subgraph_manager.add_routes(routes, loaded_modules)

    # Mock get_subgraph to return a subgraph dictionary
    mock_entry = {
        "source": "templates",
        "name": "test_graph",
        "path": "/path/to/test_graph.json",
        "info": {"node_pack": "comfyui"},
        "data": "{}"
    }

    async def mock_get_subgraph(id_val, modules):
        if id_val == "valid_id":
            return mock_entry.copy()
        return None

    subgraph_manager.get_subgraph = mock_get_subgraph

    # Find the registered handler for /global_subgraphs/{id}
    handler = None
    for route in routes:
        if route.path == "/global_subgraphs/{id}" and route.method == "GET":
            handler = route.handler
            break

    assert handler is not None

    mock_request = MagicMock()
    mock_request.match_info = {"id": "valid_id"}

    response = await handler(mock_request)
    assert response.status == 200
    body = json.loads(response.body)
    assert body["name"] == "test_graph"
    assert "path" not in body  # sanitized entry should remove internal path


@pytest.mark.asyncio
async def test_get_global_subgraph_not_found(subgraph_manager):
    routes = web.RouteTableDef()
    loaded_modules = {}
    subgraph_manager.add_routes(routes, loaded_modules)

    async def mock_get_subgraph(id_val, modules):
        return None

    subgraph_manager.get_subgraph = mock_get_subgraph

    handler = None
    for route in routes:
        if route.path == "/global_subgraphs/{id}" and route.method == "GET":
            handler = route.handler
            break

    assert handler is not None

    mock_request = MagicMock()
    mock_request.match_info = {"id": "non_existent_id"}

    response = await handler(mock_request)
    assert response.status == 404
    body = json.loads(response.body)
    assert body == {"error": "Subgraph not found"}
