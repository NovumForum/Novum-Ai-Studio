import sys
from unittest.mock import MagicMock

# Mock torch
mock_torch = MagicMock()
class DummyTensor:
    pass
mock_torch.Tensor = DummyTensor
sys.modules['torch'] = mock_torch

# Mock packaging which comfy_api.internal.api_registry might need
sys.modules['packaging'] = MagicMock()
sys.modules['packaging.version'] = MagicMock()

# Mock comfy_execution package and submodules
sys.modules['comfy_execution'] = MagicMock()
sys.modules['comfy_execution.graph_utils'] = MagicMock()
sys.modules['comfy_execution.utils'] = MagicMock()
sys.modules['comfy_execution.progress'] = MagicMock()

# Mock comfy
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.cli_args'] = MagicMock()

# Mock av package and its submodules
sys.modules['av'] = MagicMock()
sys.modules['av.container'] = MagicMock()
sys.modules['av.video'] = MagicMock()
sys.modules['av.subtitles'] = MagicMock()
sys.modules['av.subtitles.stream'] = MagicMock()

# Mock Pillow and tqdm
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.PngImagePlugin'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

# Mock numpy and numpy.dtypes
sys.modules['numpy'] = MagicMock()
sys.modules['numpy.dtypes'] = MagicMock()

from comfy_api.latest import IO

def test_number_display_enum_color():
    assert "color" in IO.NumberDisplay.__members__
    assert IO.NumberDisplay.color.value == "color"

def test_int_input_as_dict_display_mode():
    int_input = IO.Int.Input("color", default=0, min=0, max=0xFFFFFF, step=1, display_mode=IO.NumberDisplay.color)
    serialized = int_input.as_dict()
    assert serialized.get("display") == "color"

def test_float_input_as_dict_display_mode():
    float_input = IO.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider)
    serialized = float_input.as_dict()
    assert serialized.get("display") == "slider"
