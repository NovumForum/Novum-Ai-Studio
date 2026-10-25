import os
import tempfile
import folder_paths


def test_recursive_search_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create structure:
        # tmpdir/
        #   file1.txt
        #   sub1/
        #     file2.txt
        #     sub2/
        #       file3.txt
        #   excluded_dir/
        #     file4.txt
        sub1 = os.path.join(tmpdir, "sub1")
        sub2 = os.path.join(sub1, "sub2")
        excluded = os.path.join(tmpdir, "excluded_dir")
        os.makedirs(sub2)
        os.makedirs(excluded)

        f1 = os.path.join(tmpdir, "file1.txt")
        f2 = os.path.join(sub1, "file2.txt")
        f3 = os.path.join(sub2, "file3.txt")
        f4 = os.path.join(excluded, "file4.txt")

        for f in (f1, f2, f3, f4):
            with open(f, "w") as fp:
                fp.write("content")

        files, dirs = folder_paths.recursive_search(tmpdir, excluded_dir_names=["excluded_dir"])

        expected_files = [
            "file1.txt",
            os.path.join("sub1", "file2.txt"),
            os.path.join("sub1", "sub2", "file3.txt"),
        ]

        assert sorted(files) == sorted(expected_files)
        assert tmpdir in dirs
        assert sub1 in dirs
        assert sub2 in dirs
        assert excluded not in dirs


def test_recursive_search_nonexistent_directory():
    files, dirs = folder_paths.recursive_search("/path/that/does/not/exist/anywhere")
    assert files == []
    assert dirs == {}


def test_recursive_search_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        files, dirs = folder_paths.recursive_search(tmpdir)
        assert files == []
        assert tmpdir in dirs
