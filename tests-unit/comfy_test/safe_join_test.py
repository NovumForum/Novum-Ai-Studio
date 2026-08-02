import pytest
import os
import tempfile
from comfy_extras.nodes_dataset import safe_join


def test_safe_join_valid():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test basic joining
        joined = safe_join(temp_dir, "subdir")
        assert os.path.abspath(joined) == os.path.abspath(os.path.join(temp_dir, "subdir"))

        # Test with sub-subdir
        joined = safe_join(temp_dir, "subdir/nested")
        assert os.path.abspath(joined) == os.path.abspath(os.path.join(temp_dir, "subdir/nested"))

        # Test with None
        joined = safe_join(temp_dir, None)
        assert os.path.abspath(joined) == os.path.abspath(temp_dir)


def test_safe_join_path_traversal():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Traversal attempt with ..
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join(temp_dir, "../outside")

        # Absolute path attempt outside base
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join(temp_dir, "/outside")

        # Complex traversal attempt
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join(temp_dir, "subdir/../../outside")
