import pytest
import yarl
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager

pytestmark = pytest.mark.asyncio


@pytest.fixture
def model_manager():
    return ModelFileManager()


@pytest.fixture
def app(model_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_get_model_preview_path_traversal(aiohttp_client, app, tmp_path):
    model_folder = tmp_path / "models"
    model_folder.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(model_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Attempt path traversal out of model directory using percent-encoded slashes/dots
        url = yarl.URL(
            "/experiment/models/preview/checkpoints/0/sub%2f..%2f..%2fetc%2fpasswd",
            encoded=True,
        )
        response = await client.get(url)
        assert response.status == 403


async def test_get_model_preview_invalid_path_index(aiohttp_client, app, tmp_path):
    model_folder = tmp_path / "models"
    model_folder.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(model_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Non-integer path_index
        response = await client.get(
            "/experiment/models/preview/checkpoints/invalid/test.png"
        )
        assert response.status == 400


async def test_get_model_preview_out_of_bounds_path_index(
    aiohttp_client, app, tmp_path
):
    model_folder = tmp_path / "models"
    model_folder.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(model_folder)], None)},
    ):
        client = await aiohttp_client(app)

        # Out-of-bounds path_index
        response = await client.get(
            "/experiment/models/preview/checkpoints/999/test.png"
        )
        assert response.status == 404

        response_neg = await client.get(
            "/experiment/models/preview/checkpoints/-1/test.png"
        )
        assert response_neg.status == 404
