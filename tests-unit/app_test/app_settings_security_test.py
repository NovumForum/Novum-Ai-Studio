import pytest
from unittest.mock import patch
from aiohttp import web
from app.user_manager import UserManager
from app.app_settings import AppSettings


@pytest.fixture
def user_manager(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path))
    um = UserManager()
    return um


@pytest.fixture
def app(user_manager):
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app = web.Application()
    app.add_routes(routes)
    return app


@pytest.mark.asyncio
async def test_get_settings_unknown_user_raises_401(aiohttp_client, app):
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        client = await aiohttp_client(app)
        # Send request with invalid user header when multi-user is enabled
        resp = await client.get("/settings", headers={"comfy-user": "nonexistent_user"})
        assert resp.status == 401


@pytest.mark.asyncio
async def test_save_settings_unknown_user_raises_401(aiohttp_client, app):
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        client = await aiohttp_client(app)
        resp = await client.post("/settings", json={"test": "val"}, headers={"comfy-user": "nonexistent_user"})
        assert resp.status == 401


def test_get_settings_none_filepath_raises_403(user_manager, monkeypatch):
    # Mock get_request_user_filepath to return None
    monkeypatch.setattr(user_manager, "get_request_user_filepath", lambda request, file: None)
    settings = AppSettings(user_manager)

    class FakeRequest:
        pass

    with pytest.raises(web.HTTPForbidden):
        settings.get_settings(FakeRequest())


def test_save_settings_none_filepath_raises_403(user_manager, monkeypatch):
    monkeypatch.setattr(user_manager, "get_request_user_filepath", lambda request, file: None)
    settings = AppSettings(user_manager)

    class FakeRequest:
        pass

    with pytest.raises(web.HTTPForbidden):
        settings.save_settings(FakeRequest(), {"key": "value"})
