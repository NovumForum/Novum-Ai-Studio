"""Unit tests for ModelFileManager security fixes (path traversal & bounds checking in model preview route)."""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
import folder_paths
from app.model_manager import ModelFileManager


@pytest.fixture
def temp_model_folder():
    """Create temporary model folder structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create a dummy model preview
        model_file = os.path.join(checkpoint_dir, "model.safetensors")
        with open(model_file, "w") as f:
            f.write("dummy model content")

        preview_file = os.path.join(checkpoint_dir, "model.png")
        with open(preview_file, "w") as f:
            f.write("dummy image content")

        original_paths = folder_paths.folder_names_and_paths.get("checkpoints")
        folder_paths.folder_names_and_paths["checkpoints"] = ([checkpoint_dir], {".safetensors"})

        yield checkpoint_dir

        if original_paths:
            folder_paths.folder_names_and_paths["checkpoints"] = original_paths
        else:
            folder_paths.folder_names_and_paths.pop("checkpoints", None)


@pytest.mark.asyncio
async def test_get_model_preview_path_traversal_returns_403(temp_model_folder):
    """Test path traversal payload returns 403 Forbidden."""
    manager = ModelFileManager()

    routes = MagicMock()
    captured_handler = None

    def get_decorator(path):
        def decorator(fn):
            nonlocal captured_handler
            captured_handler = fn
            return fn
        return decorator

    routes.get = get_decorator
    manager.add_routes(routes)

    request = MagicMock()
    request.match_info = {
        "folder": "checkpoints",
        "path_index": "0",
        "filename": "../../../etc/passwd"
    }

    response = await captured_handler(request)
    assert response.status == 403


@pytest.mark.asyncio
async def test_get_model_preview_invalid_path_index_returns_400_or_404(temp_model_folder):
    """Test non-integer or out of bounds path_index return 400 or 404."""
    manager = ModelFileManager()

    routes = MagicMock()
    captured_handler = None

    def get_decorator(path):
        def decorator(fn):
            nonlocal captured_handler
            captured_handler = fn
            return fn
        return decorator

    routes.get = get_decorator
    manager.add_routes(routes)

    # Non-integer path_index
    request_invalid_type = MagicMock()
    request_invalid_type.match_info = {
        "folder": "checkpoints",
        "path_index": "invalid",
        "filename": "model.safetensors"
    }
    response_type = await captured_handler(request_invalid_type)
    assert response_type.status == 400

    # Out of bounds path_index
    request_out_of_bounds = MagicMock()
    request_out_of_bounds.match_info = {
        "folder": "checkpoints",
        "path_index": "99",
        "filename": "model.safetensors"
    }
    response_bounds = await captured_handler(request_out_of_bounds)
    assert response_bounds.status == 404
