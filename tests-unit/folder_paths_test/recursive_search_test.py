import os
import tempfile
import folder_paths


def test_recursive_search_empty_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        files, dirs = folder_paths.recursive_search(temp_dir)
        assert files == []
        assert temp_dir in dirs


def test_recursive_search_nonexistent_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent = os.path.join(temp_dir, "does_not_exist")
        files, dirs = folder_paths.recursive_search(non_existent)
        assert files == []
        assert dirs == {}


def test_recursive_search_flat_and_nested_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create root files
        open(os.path.join(temp_dir, "root_a.safetensors"), "w").close()
        open(os.path.join(temp_dir, "root_b.ckpt"), "w").close()

        # Create nested dirs and files
        sub1 = os.path.join(temp_dir, "sub1")
        sub2 = os.path.join(temp_dir, "sub1", "sub2")
        os.makedirs(sub2)
        open(os.path.join(sub1, "nested1.pt"), "w").close()
        open(os.path.join(sub2, "nested2.safetensors"), "w").close()

        # Excluded directory
        git_dir = os.path.join(temp_dir, ".git")
        os.makedirs(git_dir)
        open(os.path.join(git_dir, "config"), "w").close()

        files, dirs = folder_paths.recursive_search(temp_dir, excluded_dir_names=[".git"])

        expected_files = {
            "root_a.safetensors",
            "root_b.ckpt",
            os.path.join("sub1", "nested1.pt"),
            os.path.join("sub1", "sub2", "nested2.safetensors"),
        }

        assert set(files) == expected_files
        assert temp_dir in dirs
        assert sub1 in dirs
        assert sub2 in dirs
        assert git_dir not in dirs


def test_recursive_search_excluded_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        sub = os.path.join(temp_dir, "skip_me")
        os.makedirs(sub)
        open(os.path.join(sub, "secret.txt"), "w").close()
        open(os.path.join(temp_dir, "allowed.txt"), "w").close()

        files, dirs = folder_paths.recursive_search(temp_dir, excluded_dir_names=["skip_me"])
        assert files == ["allowed.txt"]
        assert sub not in dirs
