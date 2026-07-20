import pytest
import os
import tempfile
import comfy.utils


def test_safe_join_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simple join
        res = comfy.utils.safe_join(tmpdir, "subdir", "file.txt")
        expected = os.path.abspath(os.path.join(tmpdir, "subdir", "file.txt"))
        assert res == expected


def test_safe_join_create_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Join with directory creation
        subdir = "new_subdir"
        res = comfy.utils.safe_join(tmpdir, subdir, create_dir=True)
        assert os.path.isdir(res)


def test_safe_join_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Standard traversal attempt
        with pytest.raises(ValueError, match="Path traversal detected"):
            comfy.utils.safe_join(tmpdir, "..", "outside.txt")

        # More complex traversal attempt
        with pytest.raises(ValueError, match="Path traversal detected"):
            comfy.utils.safe_join(tmpdir, "subdir", "..", "..", "outside.txt")


def test_safe_join_empty_base():
    with pytest.raises(ValueError, match="Base directory cannot be empty"):
        comfy.utils.safe_join("", "file.txt")
