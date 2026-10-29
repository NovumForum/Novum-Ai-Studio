import os
import tempfile
import shutil
from unittest.mock import patch
import folder_paths


def test_get_input_subfolders_basic():
    tmp_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp_dir, "sub1", "child1"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "sub2"), exist_ok=True)

        with patch.object(folder_paths, "get_input_directory", return_value=tmp_dir):
            subfolders = folder_paths.get_input_subfolders()

        assert subfolders == ["sub1", "sub1/child1", "sub2"]
    finally:
        shutil.rmtree(tmp_dir)


def test_get_input_subfolders_trailing_slash():
    tmp_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp_dir, "alpha", "beta"), exist_ok=True)

        input_dir_trailing = tmp_dir + os.sep
        with patch.object(folder_paths, "get_input_directory", return_value=input_dir_trailing):
            subfolders = folder_paths.get_input_subfolders()

        assert subfolders == ["alpha", "alpha/beta"]
    finally:
        shutil.rmtree(tmp_dir)


def test_get_input_subfolders_empty():
    tmp_dir = tempfile.mkdtemp()
    try:
        with patch.object(folder_paths, "get_input_directory", return_value=tmp_dir):
            subfolders = folder_paths.get_input_subfolders()

        assert subfolders == []
    finally:
        shutil.rmtree(tmp_dir)


def test_get_input_subfolders_non_existent():
    non_existent = "/non/existent/path/for/comfyui/test"
    with patch.object(folder_paths, "get_input_directory", return_value=non_existent):
        subfolders = folder_paths.get_input_subfolders()

    assert subfolders == []
