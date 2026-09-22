import pytest
from aiohttp import web
import folder_paths
from app.model_manager import ModelFileManager

@pytest.fixture
def app(tmp_path):
    # Set up dummy folder structure in folder_paths
    models_dir = tmp_path / "checkpoints"
    models_dir.mkdir()

    # Create a dummy model file inside checkpoints
    dummy_model = models_dir / "test_model.ckpt"
    dummy_model.write_text("dummy content")

    # Register folder path in folder_paths.folder_names_and_paths
    original_folders = folder_paths.folder_names_and_paths.get("checkpoints")
    folder_paths.folder_names_and_paths["checkpoints"] = ([str(models_dir)], {".ckpt", ".safetensors"})

    app = web.Application()
    routes = web.RouteTableDef()
    manager = ModelFileManager()
    manager.add_routes(routes)
    app.add_routes(routes)

    yield app

    # Cleanup
    if original_folders is not None:
        folder_paths.folder_names_and_paths["checkpoints"] = original_folders
    else:
        folder_paths.folder_names_and_paths.pop("checkpoints", None)

@pytest.mark.asyncio
async def test_get_model_preview_path_traversal(app, aiohttp_client):
    client = await aiohttp_client(app)

    # Attempt path traversal out of model directory using URL encoding
    resp = await client.get("/experiment/models/preview/checkpoints/0/..%2F..%2F..%2F..%2Fetc%2Fpasswd")
    assert resp.status == 403

@pytest.mark.asyncio
async def test_get_model_preview_invalid_path_index_type(app, aiohttp_client):
    client = await aiohttp_client(app)

    # Non-integer path_index
    resp = await client.get("/experiment/models/preview/checkpoints/invalid/test_model.ckpt")
    assert resp.status == 400

@pytest.mark.asyncio
async def test_get_model_preview_out_of_bounds_path_index(app, aiohttp_client):
    client = await aiohttp_client(app)

    # Out of bounds path_index
    resp = await client.get("/experiment/models/preview/checkpoints/99/test_model.ckpt")
    assert resp.status == 404
