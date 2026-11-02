import pytest
from unittest.mock import MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    server.send_sync = MagicMock()
    return server


@pytest.fixture
def app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()


async def test_subscribe_logs_valid_enabled_true(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.patch("/logs/subscribe", json={"clientId": "client-123", "enabled": True})
    assert resp.status == 200


async def test_subscribe_logs_valid_enabled_false(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.patch("/logs/subscribe", json={"clientId": "client-123", "enabled": False})
    assert resp.status == 200


async def test_subscribe_logs_invalid_json(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.patch(
        "/logs/subscribe",
        data="not-a-valid-json",
        headers={"Content-Type": "application/json"}
    )
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid JSON"}


async def test_subscribe_logs_invalid_payload_not_dict(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.patch("/logs/subscribe", json=["clientId", "client-123"])
    assert resp.status == 400
    data = await resp.json()
    assert data == {"error": "Invalid payload format"}


async def test_subscribe_logs_missing_parameters(aiohttp_client, app):
    client = await aiohttp_client(app)

    # Missing clientId
    resp1 = await client.patch("/logs/subscribe", json={"enabled": True})
    assert resp1.status == 400
    assert await resp1.json() == {"error": "Invalid clientId or enabled parameter"}

    # Missing enabled
    resp2 = await client.patch("/logs/subscribe", json={"clientId": "client-123"})
    assert resp2.status == 400
    assert await resp2.json() == {"error": "Invalid clientId or enabled parameter"}


async def test_subscribe_logs_invalid_parameter_types(aiohttp_client, app):
    client = await aiohttp_client(app)

    # Empty string clientId
    resp1 = await client.patch("/logs/subscribe", json={"clientId": "   ", "enabled": True})
    assert resp1.status == 400
    assert await resp1.json() == {"error": "Invalid clientId or enabled parameter"}

    # Non-string clientId (e.g. integer)
    resp2 = await client.patch("/logs/subscribe", json={"clientId": 12345, "enabled": True})
    assert resp2.status == 400
    assert await resp2.json() == {"error": "Invalid clientId or enabled parameter"}

    # Non-boolean enabled (e.g. string "true")
    resp3 = await client.patch("/logs/subscribe", json={"clientId": "client-123", "enabled": "true"})
    assert resp3.status == 400
    assert await resp3.json() == {"error": "Invalid clientId or enabled parameter"}

    # Non-boolean enabled (e.g. integer 1)
    resp4 = await client.patch("/logs/subscribe", json={"clientId": "client-123", "enabled": 1})
    assert resp4.status == 400
    assert await resp4.json() == {"error": "Invalid clientId or enabled parameter"}
