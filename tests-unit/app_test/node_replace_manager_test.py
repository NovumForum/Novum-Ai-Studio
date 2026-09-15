from app.node_replace_manager import NodeReplaceManager
from comfy_api.latest._io_public import NodeReplace
from unittest.mock import patch, MagicMock

import torch
# Mock torch.cuda before importing anything that depends on it
torch.cuda.is_available = MagicMock(return_value=False)
torch.cuda.current_device = MagicMock(return_value=0)
torch.cuda.get_device_properties = MagicMock(return_value=MagicMock(total_memory=1024*1024*1024))
torch.cuda.memory_stats = MagicMock(return_value={'reserved_bytes.all.current': 0})
torch.cuda.mem_get_info = MagicMock(return_value=(1024*1024*1024, 1024*1024*1024))

def test_node_replace_manager_updates_meta_title():

    with patch("app.node_replace_manager.nodes") as mock_nodes:
        # Node must not be in NODE_CLASS_MAPPINGS to be replaced
        mock_nodes.NODE_CLASS_MAPPINGS = {"NewNode": type("NewNode", (), {})}
        mock_nodes.NODE_DISPLAY_NAME_MAPPINGS = {"NewNode": "New Node Display Name"}

        manager = NodeReplaceManager()
        manager.register(NodeReplace(old_node_id="OldNode", new_node_id="NewNode", input_mapping=[], output_mapping=[]))

        prompt = {
            "1": {
                "class_type": "OldNode",
                "inputs": {},
                "_meta": {"title": "Old Node"}
            },
            "2": {
                "class_type": "OldNode",
                "inputs": {},
                "_meta": {"title": "Custom Title"}
            },
            "3": {
                "class_type": "OldNode",
                "inputs": {}
            }
        }

        manager.apply_replacements(prompt)

        assert prompt["1"]["class_type"] == "NewNode"
        assert prompt["1"]["_meta"]["title"] == "New Node Display Name"

        assert prompt["2"]["class_type"] == "NewNode"
        assert prompt["2"]["_meta"]["title"] == "New Node Display Name"

        assert prompt["3"]["class_type"] == "NewNode"
        assert "_meta" in prompt["3"]
        assert prompt["3"]["_meta"] == {"title": "New Node Display Name"}
