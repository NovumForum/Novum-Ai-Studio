import os
import sys

# Mock folder_paths
import folder_paths

folder_paths.input_directory = os.path.abspath("input")
folder_paths.output_directory = os.path.abspath("output")
folder_paths.temp_directory = os.path.abspath("temp")

def test_path_traversal(path):
    print(f"Testing path: {path}")
    resolved = folder_paths.get_annotated_filepath(path)
    print(f"Resolved path: {resolved}")

    # Check if it's outside input_directory
    if not os.path.abspath(resolved).startswith(os.path.abspath(folder_paths.input_directory)):
        print("VULNERABLE: Path is outside input directory!")
    else:
        print("SAFE: Path is inside input directory (or seems so).")

test_path_traversal("../../etc/passwd")
test_path_traversal("/etc/passwd")
