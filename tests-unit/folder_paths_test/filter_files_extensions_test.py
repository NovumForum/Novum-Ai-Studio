import os
import folder_paths


def test_filter_files_extensions_basic():
    files = ["model.safetensors", "model.ckpt", "image.png", "notes.txt"]
    exts = [".safetensors", ".ckpt"]
    assert folder_paths.filter_files_extensions(files, exts) == ["model.ckpt", "model.safetensors"]


def test_filter_files_extensions_case_insensitivity():
    files = ["model1.SAFETENSORS", "model2.Ckpt", "model3.pt", "other.bin"]
    exts = [".safetensors", ".ckpt", ".PT"]
    assert folder_paths.filter_files_extensions(files, exts) == ["model1.SAFETENSORS", "model2.Ckpt", "model3.pt"]


def test_filter_files_extensions_empty_extensions():
    files = ["c.txt", "a.png", "b.jpg"]
    assert folder_paths.filter_files_extensions(files, []) == ["a.png", "b.jpg", "c.txt"]
    assert folder_paths.filter_files_extensions(files, set()) == ["a.png", "b.jpg", "c.txt"]


def test_filter_files_extensions_empty_files():
    assert folder_paths.filter_files_extensions([], [".safetensors"]) == []
    assert folder_paths.filter_files_extensions([], []) == []


def test_filter_files_extensions_subdirectories_and_leading_dots():
    files = [
        "sub/folder/model.safetensors",
        "sub/folder/.hidden",
        "sub/folder/file_without_ext",
        "another.dir/file",
        "another.dir/valid.pt",
    ]
    exts = [".safetensors", ".pt"]
    assert folder_paths.filter_files_extensions(files, exts) == [
        "another.dir/valid.pt",
        "sub/folder/model.safetensors",
    ]


def test_filter_files_extensions_exact_os_path_splitext_equivalence():
    files = [
        "file1.txt",
        "file2.PNG",
        "file3.jpg",
        "dir.ext/file",
        ".hidden",
        "sub/.hidden.ext",
        "a/b/c.safetensors",
        "noext",
        "",
    ]
    exts = [".txt", ".png", ".ext", ".safetensors"]

    # Reference behavior using os.path.splitext
    expected = sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in exts or len(exts) == 0, files)))
    actual = folder_paths.filter_files_extensions(files, exts)
    assert actual == expected
