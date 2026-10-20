import pytest
import os
from aiohttp import web
from app.user_manager import UserManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    def mock_get_request_user_filepath(req, file, **kwargs):
        if req.headers.get("comfy-user") == "invalid_user":
            raise KeyError("Unknown user: invalid_user")
        if file:
            return os.path.join(tmp_path, file)
        return str(tmp_path)
    um.get_request_user_filepath = mock_get_request_user_filepath
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_delete_userdata_file(aiohttp_client, app, tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello")

    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/test_file.txt")
    assert resp.status == 204
    assert not file_path.exists()


async def test_delete_userdata_directory(aiohttp_client, app, tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    (dir_path / "sub_file.txt").write_text("sub content")

    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/test_dir")
    assert resp.status == 204
    assert not dir_path.exists()


async def test_delete_userdata_nonexistent(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/nonexistent.txt")
    assert resp.status == 404


async def test_delete_userdata_invalid_user(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/some_file.txt", headers={"comfy-user": "invalid_user"})
    assert resp.status == 403
