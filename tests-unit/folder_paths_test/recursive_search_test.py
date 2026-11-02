import os
import tempfile
import folder_paths


def test_recursive_search_empty_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        files, dirs = folder_paths.recursive_search(temp_dir)
        assert files == []
        assert temp_dir in dirs


def test_recursive_search_nonexistent_dir():
    nonexistent = "/nonexistent_path_that_does_not_exist_12345"
    files, dirs = folder_paths.recursive_search(nonexistent)
    assert files == []
    assert dirs == {}


def test_recursive_search_flat_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1 = os.path.join(temp_dir, "model_a.safetensors")
        file2 = os.path.join(temp_dir, "model_b.ckpt")
        open(file1, "w").close()
        open(file2, "w").close()

        files, dirs = folder_paths.recursive_search(temp_dir)
        assert sorted(files) == ["model_a.safetensors", "model_b.ckpt"]
        assert temp_dir in dirs
        assert len(dirs) == 1


def test_recursive_search_nested_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested folders
        sub1 = os.path.join(temp_dir, "category1")
        sub2 = os.path.join(temp_dir, "category2", "subcat")
        os.makedirs(sub1)
        os.makedirs(sub2)

        # Create files
        open(os.path.join(temp_dir, "root.txt"), "w").close()
        open(os.path.join(sub1, "sub1_file.safetensors"), "w").close()
        open(os.path.join(sub2, "deep_file.safetensors"), "w").close()

        files, dirs = folder_paths.recursive_search(temp_dir)
        expected_files = [
            "root.txt",
            os.path.join("category1", "sub1_file.safetensors"),
            os.path.join("category2", "subcat", "deep_file.safetensors"),
        ]
        assert sorted(files) == sorted(expected_files)
        assert temp_dir in dirs
        assert sub1 in dirs
        assert os.path.join(temp_dir, "category2") in dirs
        assert sub2 in dirs


def test_recursive_search_excluded_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        git_dir = os.path.join(temp_dir, ".git")
        os.makedirs(git_dir)
        open(os.path.join(git_dir, "config"), "w").close()

        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir)
        open(os.path.join(model_dir, "model.ckpt"), "w").close()

        files, dirs = folder_paths.recursive_search(temp_dir, excluded_dir_names=[".git"])
        assert files == [os.path.join("models", "model.ckpt")]
        assert git_dir not in dirs
        assert model_dir in dirs
