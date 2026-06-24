import sys
from unittest.mock import MagicMock

# pylint: disable=import-error, wrong-import-position, import-outside-toplevel

# Define a mock for torch
mock_torch = MagicMock()
mock_torch.__version__ = "2.1.0"
mock_torch.cuda.is_available.return_value = False
mock_torch.device.return_value = "cpu"
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda

# Mock other heavy dependencies to avoid deep imports triggering hardware checks
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()
sys.modules['comfy_api_nodes.util'] = MagicMock()
sys.modules['comfy_extras.nodes_images'] = MagicMock()

from comfy_api_nodes.nodes_recraft import RecraftColorNode
from comfy_api_nodes.apis.recraft import RecraftColorChain

def test_recraft_color_node_hex_to_rgb():
    # Test white
    result = RecraftColorNode.execute(color="#ffffff")
    chain = result.args[0]
    assert len(chain.colors) == 1
    assert chain.colors[0].color == [255, 255, 255]

    # Test black
    result = RecraftColorNode.execute(color="#000000")
    chain = result.args[0]
    assert len(chain.colors) == 1
    assert chain.colors[0].color == [0, 0, 0]

    # Test specific color
    result = RecraftColorNode.execute(color="#ff5733")
    chain = result.args[0]
    assert len(chain.colors) == 1
    assert chain.colors[0].color == [255, 87, 51]

    # Test without #
    result = RecraftColorNode.execute(color="00ff00")
    chain = result.args[0]
    assert len(chain.colors) == 1
    assert chain.colors[0].color == [0, 255, 0]

def test_recraft_color_node_chaining():
    # Start with red
    red_chain = RecraftColorChain()
    from comfy_api_nodes.apis.recraft import RecraftColor
    red_chain.add(RecraftColor(255, 0, 0))

    # Add green via node
    result = RecraftColorNode.execute(color="#00ff00", recraft_color=red_chain)
    chain = result.args[0]

    assert len(chain.colors) == 2
    assert chain.colors[0].color == [255, 0, 0]
    assert chain.colors[1].color == [0, 255, 0]
