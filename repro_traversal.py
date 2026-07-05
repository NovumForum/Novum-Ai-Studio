import os
import sys

# Mock folder_paths
class MockFolderPaths:
    def get_input_directory(self):
        return os.path.abspath("input")
    def get_output_directory(self):
        return os.path.abspath("output")

folder_paths = MockFolderPaths()

def test_path_traversal(folder):
    input_dir = folder_paths.get_input_directory()
    # This is what the code currently does
    sub_input_dir = os.path.join(input_dir, folder)
    print(f"Input: {folder}")
    print(f"Result: {sub_input_dir}")
    print(f"Absolute: {os.path.abspath(sub_input_dir)}")

    # Secure version
    def safe_join(base, path):
        base = os.path.abspath(base)
        joined = os.path.abspath(os.path.join(base, path))
        if os.path.commonpath([base, joined]) != base:
            raise ValueError(f"Path traversal detected: {path}")
        return joined

    try:
        secure_path = safe_join(input_dir, folder)
        print(f"Secure Result: {secure_path}")
    except ValueError as e:
        print(f"Secure Result: Caught expected error: {e}")
    print("-" * 20)

if __name__ == "__main__":
    if not os.path.exists("input"): os.makedirs("input")
    test_path_traversal("dataset1")
    test_path_traversal("../outside")
    test_path_traversal("/etc/passwd")
