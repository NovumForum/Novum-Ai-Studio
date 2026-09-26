import pytest
from unittest.mock import MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes


@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server


@pytest.fixture
def internal_routes_app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()


class TestInternalLogsSubscribeEndpoint:
    @pytest.mark.asyncio
    async def test_subscribe_logs_success(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": "client123", "enabled": True}
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unsubscribe_logs_success(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": "client123", "enabled": False}
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_subscribe_logs_invalid_json(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            data="not a json string",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_subscribe_logs_not_json_object(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json=["item1", "item2"]
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_subscribe_logs_missing_client_id(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"enabled": True}
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_subscribe_logs_non_string_client_id(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": 12345, "enabled": True}
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_subscribe_logs_empty_client_id(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": "", "enabled": True}
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_subscribe_logs_missing_enabled(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": "client123"}
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_subscribe_logs_non_boolean_enabled(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.patch(
            "/logs/subscribe",
            json={"clientId": "client123", "enabled": "true"}
        )
        assert resp.status == 400


class TestInternalFilesEndpoint:
    @pytest.mark.asyncio
    async def test_get_files_invalid_directory_type(self, aiohttp_client, internal_routes_app):
        client = await aiohttp_client(internal_routes_app)
        resp = await client.get("/files/invalid_type")
        assert resp.status == 400
