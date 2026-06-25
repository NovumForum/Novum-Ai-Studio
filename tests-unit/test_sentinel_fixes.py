import os
import pytest
import sys
from unittest.mock import MagicMock

# Mock folder_paths before importing nodes_dataset
mock_folder_paths = MagicMock()
mock_folder_paths.get_input_directory.return_value = "/tmp/input"
mock_folder_paths.get_output_directory.return_value = "/tmp/output"
sys.modules["folder_paths"] = mock_folder_paths

# Mock node_helpers
sys.modules["node_helpers"] = MagicMock()

# Mock comfy_api.latest
mock_comfy_api = MagicMock()
sys.modules["comfy_api"] = mock_comfy_api
sys.modules["comfy_api.latest"] = mock_comfy_api

from comfy_extras.nodes_dataset import safe_join

def test_safe_join_valid():
    base = "/tmp/input"
    path = "images/cat.png"
    # Note: os.path.abspath will resolve /tmp/input/images/cat.png
    # In sandbox, it might be /app/tmp/input/images/cat.png if /tmp is not at root,
    # but abspath handles it.
    expected = os.path.abspath(os.path.join(base, path))
    assert safe_join(base, path) == expected

def test_safe_join_traversal():
    base = "/tmp/input"
    path = "../../../etc/passwd"
    with pytest.raises(ValueError, match="Security error: Path traversal detected"):
        safe_join(base, path)

def test_safe_join_encoded_traversal():
    base = "/tmp/input"
    path = "subdir/../../etc/passwd"
    with pytest.raises(ValueError, match="Security error: Path traversal detected"):
        safe_join(base, path)

if __name__ == "__main__":
    # Manual run if pytest is not used
    try:
        test_safe_join_valid()
        print("test_safe_join_valid passed")
        test_safe_join_traversal()
        print("test_safe_join_traversal passed")
        test_safe_join_encoded_traversal()
        print("test_safe_join_encoded_traversal passed")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
