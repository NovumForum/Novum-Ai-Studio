import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock dependencies robustly
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['numpy.dtypes'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.PngImagePlugin'] = MagicMock()
sys.modules['typing_extensions'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['tqdm.auto'] = MagicMock()
sys.modules['einops'] = MagicMock()
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['packaging'] = MagicMock()
sys.modules['packaging.version'] = MagicMock()
sys.modules['comfy_aimdo'] = MagicMock()
sys.modules['comfy_kitchen'] = MagicMock()
sys.modules['av'] = MagicMock()

import comfy.utils
import folder_paths

class TestSafeJoin(unittest.TestCase):
    def test_safe_join_valid(self):
        base = "/tmp/comfy_base"
        os.makedirs(base, exist_ok=True)
        path = comfy.utils.safe_join(base, "sub", "file.txt")
        self.assertEqual(path, os.path.abspath(os.path.join(base, "sub", "file.txt")))

    def test_safe_join_traversal(self):
        base = "/tmp/comfy_base"
        os.makedirs(base, exist_ok=True)
        path = comfy.utils.safe_join(base, "..", "outside.txt")
        self.assertIsNone(path)

    def test_safe_join_absolute_traversal(self):
        base = "/tmp/comfy_base"
        os.makedirs(base, exist_ok=True)
        path = comfy.utils.safe_join(base, "/etc/passwd")
        self.assertIsNone(path)

class TestDatasetNodesSecurity(unittest.TestCase):
    def setUp(self):
        folder_paths.get_input_directory = MagicMock(return_value="/input")
        folder_paths.get_output_directory = MagicMock(return_value="/output")

    def test_load_image_dataset_traversal(self):
        from comfy_extras.nodes_dataset import LoadImageDataSetFromFolderNode
        with self.assertRaises(ValueError):
            LoadImageDataSetFromFolderNode.execute(folder="../etc")

    def test_save_image_dataset_traversal(self):
        from comfy_extras.nodes_dataset import SaveImageDataSetToFolderNode
        with self.assertRaises(ValueError):
            SaveImageDataSetToFolderNode.execute(["img1"], ["../forbidden"], ["prefix"])

if __name__ == "__main__":
    unittest.main()
