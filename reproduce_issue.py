import os
import sys
import torch

# Mock folder_paths
class MockFolderPaths:
    def get_input_directory(self):
        return os.path.abspath("input")
    def get_output_directory(self):
        return os.path.abspath("output")
    def get_input_subfolders(self):
        return []

sys.modules['folder_paths'] = MockFolderPaths()

# Mock node_helpers
class MockNodeHelpers:
    def pillow(self, fn, arg):
        return fn(arg)
sys.modules['node_helpers'] = MockNodeHelpers()

# Mock comfy_api
class MockIO:
    class ComfyNode: pass
    class Schema:
        def __init__(self, **kwargs): pass
    class Combo:
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Image:
        class Output:
            def __init__(self, *args, **kwargs): pass
        class Input:
            def __init__(self, *args, **kwargs): pass
    class String:
        class Output:
            def __init__(self, *args, **kwargs): pass
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Int:
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Float:
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Vae:
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Clip:
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Latent:
        class Output:
            def __init__(self, *args, **kwargs): pass
        class Input:
            def __init__(self, *args, **kwargs): pass
    class Conditioning:
        class Output:
            def __init__(self, *args, **kwargs): pass
        class Input:
            def __init__(self, *args, **kwargs): pass
    class NodeOutput:
        def __init__(self, *args, **kwargs): pass

class MockComfyExtension: pass

sys.modules['comfy_api.latest'] = type('module', (), {'io': MockIO, 'ComfyExtension': MockComfyExtension})

from comfy_extras.nodes_dataset import LoadImageDataSetFromFolderNode, SaveImageDataSetToFolderNode

def test_path_traversal():
    print("Testing path traversal...")

    # Create dummy directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("outside", exist_ok=True)
    with open("outside/secret.txt", "w") as f:
        f.write("sensitive data")

    node = LoadImageDataSetFromFolderNode()
    try:
        # This should fail if safe_join is working
        node.execute(folder="../../outside")
        print("VULNERABLE: LoadImageDataSetFromFolderNode allowed access outside input directory")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

    save_node = SaveImageDataSetToFolderNode()
    try:
        save_node.execute(images=[torch.zeros((1, 64, 64, 3))], folder_name=["../../outside"], filename_prefix=["test"])
        if os.path.exists("outside/test_00000.png") or os.path.isdir("outside/dataset"):
             print("VULNERABLE: SaveImageDataSetToFolderNode allowed writing outside output directory")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_path_traversal()
