import sys
import os
import unittest.mock

# Mock necessary modules that might fail to import
sys.modules['scipy'] = unittest.mock.MagicMock()
sys.modules['scipy.ndimage'] = unittest.mock.MagicMock()

# Mocking ComfyUI core modules that might have heavy dependencies
sys.path.insert(0, os.path.abspath('.'))

from comfy_extras.nodes_mask import ImageColorToMask
from comfy_api.latest import IO

def verify():
    schema = ImageColorToMask.GET_SCHEMA()
    info = schema.get_v1_info(ImageColorToMask)

    color_input = info.input['required']['color']
    print(f"Color input info: {color_input}")

    display_mode = color_input[1].get('display')
    print(f"Display mode: {display_mode}")

    if display_mode == 'color':
        print("Verification SUCCESS: display mode is 'color'")
    else:
        print(f"Verification FAILED: expected 'color', got '{display_mode}'")
        sys.exit(1)

if __name__ == "__main__":
    verify()
