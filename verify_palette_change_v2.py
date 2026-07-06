import sys
import os
import unittest.mock

# Mock torch before anything else to avoid CUDA checks
mock_torch = unittest.mock.MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = unittest.mock.MagicMock()

# Mock other problematic modules
sys.modules['scipy'] = unittest.mock.MagicMock()
sys.modules['scipy.ndimage'] = unittest.mock.MagicMock()
sys.modules['comfy_aimdo'] = unittest.mock.MagicMock()
sys.modules['comfy_kitchen'] = unittest.mock.MagicMock()

# Mocking ComfyUI core modules
sys.path.insert(0, os.path.abspath('.'))

# Now we can try to import the necessary bits
from comfy_api.latest import IO

def verify():
    # We want to check the serialization of ImageColorToMask
    # But since we mocked torch, we can't easily import nodes that use it without more mocks
    # Let's check the enum and a manual instantiation of IO.Int.Input

    print(f"NumberDisplay.color: {IO.NumberDisplay.color}")
    if IO.NumberDisplay.color != 'color':
        print(f"FAILED: NumberDisplay.color is {IO.NumberDisplay.color}")
        sys.exit(1)

    color_input = IO.Int.Input("color", default=0, min=0, max=0xFFFFFF, step=1, display_mode=IO.NumberDisplay.color)
    as_dict = color_input.as_dict()
    print(f"Input as_dict: {as_dict}")

    if as_dict.get('display') == 'color':
        print("Verification SUCCESS: display mode is correctly serialized to 'color'")
    else:
        print(f"Verification FAILED: expected 'display': 'color', got {as_dict}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
