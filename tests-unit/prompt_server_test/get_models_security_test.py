import pytest
from unittest.mock import patch, MagicMock
from aiohttp import web
import folder_paths


@pytest.fixture
def app():
    app = web.Application()

    async def get_models(request):
        folder = request.match_info.get("folder", None)
        import os
        if not folder or "/" in folder or "\\" in folder or ".." in folder or os.path.basename(folder) != folder:
            return web.Response(status=404)
        if folder not in folder_paths.folder_names_and_paths:
            return web.Response(status=404)
        files = folder_paths.get_filename_list(folder)
        return web.json_response(files)

    app.router.add_get("/models/{folder:.*}", get_models)
    return app


@pytest.mark.asyncio
async def test_get_models_valid_folder(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        with patch.object(folder_paths, "get_filename_list", return_value=["model1.safetensors", "model2.safetensors"]):
            resp = await client.get("/models/checkpoints")
            assert resp.status == 200
            data = await resp.json()
            assert data == ["model1.safetensors", "model2.safetensors"]


@pytest.mark.asyncio
async def test_get_models_path_traversal_dotdot(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        resp = await client.get("/models/..")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_get_models_path_traversal_subfolder(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        resp = await client.get("/models/checkpoints/..")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_get_models_invalid_folder(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch.object(folder_paths, "folder_names_and_paths", {"checkpoints": (["/fake/path"], {})}):
        resp = await client.get("/models/nonexistent_folder")
        assert resp.status == 404
