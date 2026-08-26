import sys
from unittest.mock import MagicMock

# Create a mock module structure for missing third-party packages in lightweight test runner environment
class MockModule(MagicMock):
    def __getattr__(self, name):
        return MagicMock()

mock_mod = MockModule()
sys.modules['kornia'] = mock_mod
sys.modules['kornia.filters'] = mock_mod
sys.modules['comfy.model_management'] = mock_mod
sys.modules['comfy.cli_args'] = mock_mod
sys.modules['torch'] = mock_mod
sys.modules['numpy'] = mock_mod
sys.modules['PIL'] = mock_mod
sys.modules['PIL.Image'] = mock_mod
sys.modules['PIL.PngImagePlugin'] = mock_mod
sys.modules['av'] = mock_mod
sys.modules['av.container'] = mock_mod
sys.modules['av.subtitles'] = mock_mod
sys.modules['av.subtitles.stream'] = mock_mod
sys.modules['tqdm'] = mock_mod

import pytest
from comfy_extras.nodes_canny import Canny


def test_canny_schema_metadata():
    schema = Canny.define_schema()

    assert schema.node_id == "Canny"
    assert schema.display_name == "Canny Edge Detector"
    assert "Detects sharp edges" in schema.description
    assert "ControlNet guidance" in schema.description

    assert "canny edge detector" in schema.search_aliases
    assert "controlnet preprocessor" in schema.search_aliases
    assert "sketch" in schema.search_aliases
    assert "edges" in schema.search_aliases

    # Verify input tooltips
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_by_id
    assert inputs_by_id["image"].tooltip == "Input image tensor to perform edge detection on."

    assert "low_threshold" in inputs_by_id
    assert "Lower intensity gradient threshold" in inputs_by_id["low_threshold"].tooltip

    assert "high_threshold" in inputs_by_id
    assert "Upper intensity gradient threshold" in inputs_by_id["high_threshold"].tooltip

    # Verify output tooltips
    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "Output edge map image (RGB format)."
