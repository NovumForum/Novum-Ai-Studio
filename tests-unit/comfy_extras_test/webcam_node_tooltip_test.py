import sys
import unittest
from unittest.mock import MagicMock

# Mock dependencies before imports
comfy_mock = MagicMock()
sys.modules["comfy"] = comfy_mock
sys.modules["comfy.cli_args"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()

for mod in [
    "torch", "torch.nn", "torch.nn.functional", "PIL", "PIL.Image", "PIL.ImageOps",
    "PIL.ImageSequence", "scipy", "scipy.ndimage"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

class MockLoadImage:
    def load_image(self, *args, **kwargs):
        pass
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        pass

nodes_mock = MagicMock()
nodes_mock.LoadImage = MockLoadImage
nodes_mock.MAX_RESOLUTION = 16384
sys.modules["nodes"] = nodes_mock
sys.modules["folder_paths"] = MagicMock()

from comfy_extras.nodes_webcam import WebcamCapture

class TestWebcamCaptureUX(unittest.TestCase):
    def test_webcam_capture_metadata_and_tooltips(self):
        self.assertTrue(hasattr(WebcamCapture, "DESCRIPTION"))
        self.assertIn("webcam", WebcamCapture.DESCRIPTION.lower())
        self.assertTrue(hasattr(WebcamCapture, "SEARCH_ALIASES"))
        self.assertIn("camera feed", WebcamCapture.SEARCH_ALIASES)

        input_types = WebcamCapture.INPUT_TYPES()
        required_inputs = input_types.get("required", {})

        self.assertIn("image", required_inputs)
        self.assertIn("width", required_inputs)
        self.assertIn("height", required_inputs)
        self.assertIn("capture_on_queue", required_inputs)

        self.assertIn("tooltip", required_inputs["image"][1])
        self.assertIn("tooltip", required_inputs["width"][1])
        self.assertIn("tooltip", required_inputs["height"][1])
        self.assertIn("tooltip", required_inputs["capture_on_queue"][1])

if __name__ == "__main__":
    unittest.main()
