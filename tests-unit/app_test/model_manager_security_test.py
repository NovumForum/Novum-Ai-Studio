import pytest
from unittest.mock import patch
from aiohttp import web
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
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(models_dir)], None)},
    ):
        client = await aiohttp_client(app)

        # Attempt path traversal using URL-encoded slashes/dots (%2e%2e%2f)
        response = await client.get(
            "/experiment/models/preview/checkpoints/0/%2e%2e%2fsecret.txt"
        )
        assert response.status == 403
        assert await response.text() == "Access denied"


async def test_get_model_preview_invalid_path_index_type(aiohttp_client, app, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(models_dir)], None)},
    ):
        client = await aiohttp_client(app)

        # Pass non-integer path_index
        response = await client.get(
            "/experiment/models/preview/checkpoints/invalid/model.safetensors"
        )
        assert response.status == 400
        assert await response.text() == "Invalid path_index"


async def test_get_model_preview_out_of_bounds_path_index(aiohttp_client, app, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with patch(
        "folder_paths.folder_names_and_paths",
        {"checkpoints": ([str(models_dir)], None)},
    ):
        client = await aiohttp_client(app)

        # Index out of range
        response = await client.get(
            "/experiment/models/preview/checkpoints/99/model.safetensors"
        )
        assert response.status == 404

        # Negative index out of range
        response_neg = await client.get(
            "/experiment/models/preview/checkpoints/-1/model.safetensors"
        )
        assert response_neg.status == 404
