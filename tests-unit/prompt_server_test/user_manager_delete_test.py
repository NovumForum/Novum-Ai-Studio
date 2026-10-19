import os
import pytest
from aiohttp import web
from app.user_manager import UserManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    um.get_request_user_filepath = lambda req, file, **kwargs: (
        os.path.join(tmp_path, file) if file else str(tmp_path)
    )
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_delete_userdata_file(aiohttp_client, app, tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("hello world")

    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/test_file.txt")
    assert resp.status == 204
    assert not test_file.exists()


async def test_delete_userdata_directory(aiohttp_client, app, tmp_path):
    test_folder = tmp_path / "test_folder"
    test_folder.mkdir()
    (test_folder / "file.txt").write_text("inner file")

    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/test_folder")
    assert resp.status == 204
    assert not test_folder.exists()


async def test_delete_userdata_nonexistent(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    resp = await client.delete("/userdata/nonexistent.txt")
    assert resp.status == 404
