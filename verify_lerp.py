import sys
from unittest.mock import MagicMock

# Mock CUDA/GPU
torch = MagicMock()
torch.cuda.is_available.return_value = False
torch.device.side_effect = lambda x: x
torch.lerp = lambda a, b, weight: a + (b - a) * weight
sys.modules['torch'] = torch

# Mock other heavy deps
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['comfy_aimdo'] = MagicMock()
sys.modules['comfy_aimdo.torch'] = MagicMock()
sys.modules['comfy_kitchen'] = MagicMock()
sys.modules['comfy_kitchen.torch'] = MagicMock()

import torch as torch_mock
from comfy.samplers import cfg_function

def test_cfg_function_lerp():
    cond_pred = torch.randn(1, 4, 64, 64)
    uncond_pred = torch.randn(1, 4, 64, 64)
    cond_scale = 7.5

    # We need to make sure cfg_function uses torch.lerp
    # Since we mocked torch.lerp, we can check if it's called
    torch_mock.lerp = MagicMock(side_effect=lambda a, b, weight: a + (b - a) * weight)

    res = cfg_function(None, cond_pred, uncond_pred, cond_scale, None, None)

    torch_mock.lerp.assert_called_once_with(uncond_pred, cond_pred, cond_scale)
    print("CFG function lerp test passed!")

if __name__ == "__main__":
    test_cfg_function_lerp()
