import pytest
import sys
from unittest.mock import MagicMock

# Mock out everything that causes comfy_api/folder_paths imports to fail
sys.modules['comfy.cli_args'] = MagicMock()
import torch
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.model_management'].get_torch_device = MagicMock(return_value=torch.device('cpu'))

sys.modules['comfy_execution'] = MagicMock()
sys.modules['comfy_execution.graph_utils'] = MagicMock()
sys.modules['comfy_execution.graph_utils'].is_link = lambda x: isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], int)

sys.modules['nodes'] = MagicMock()

from app.node_replace_manager import NodeReplaceManager, NodeStruct
import nodes

class NodeReplace:
    def __init__(self, old_node_id, new_node_id, input_mapping=None, output_mapping=None):
        self.old_node_id = old_node_id
        self.new_node_id = new_node_id
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def as_dict(self):
        return {
            "old_node_id": self.old_node_id,
            "new_node_id": self.new_node_id,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping
        }

# Remock the properties
nodes.NODE_CLASS_MAPPINGS = {}
nodes.NODE_DISPLAY_NAME_MAPPINGS = {}

@pytest.fixture
def manager():
    return NodeReplaceManager()

def test_apply_replacements(manager):
    nodes.NODE_CLASS_MAPPINGS["NewNode"] = type("NewNode", (), {})
    nodes.NODE_DISPLAY_NAME_MAPPINGS["NewNode"] = "My New Node"

    manager.register(NodeReplace(old_node_id="OldNode", new_node_id="NewNode"))

    prompt = {
        "1": {
            "class_type": "OldNode",
            "inputs": {"a": 1},
            "_meta": {"title": "Old Node"}
        }
    }

    manager.apply_replacements(prompt)

    assert prompt["1"]["class_type"] == "NewNode"
    assert prompt["1"]["_meta"].get("display_name") == "My New Node"

def test_apply_replacements_display_name(manager):
    nodes.NODE_CLASS_MAPPINGS["NewNode"] = type("NewNode", (), {})
    nodes.NODE_DISPLAY_NAME_MAPPINGS["NewNode"] = "My New Node"

    manager.register(NodeReplace(old_node_id="OldNode", new_node_id="NewNode"))

    prompt = {
        "1": {
            "class_type": "OldNode",
            "inputs": {"a": 1},
            "_meta": {"title": "Old Node", "display_name": "Old Node"}
        }
    }

    manager.apply_replacements(prompt)

    assert prompt["1"]["class_type"] == "NewNode"
    assert prompt["1"]["_meta"].get("display_name") == "My New Node"

def test_apply_replacements_no_meta_no_problem(manager):
    nodes.NODE_CLASS_MAPPINGS["NewNode"] = type("NewNode", (), {})
    nodes.NODE_DISPLAY_NAME_MAPPINGS["NewNode"] = "My New Node"

    manager.register(NodeReplace(old_node_id="OldNode", new_node_id="NewNode"))

    prompt = {
        "1": {
            "class_type": "OldNode",
            "inputs": {"a": 1},
            # No _meta
        }
    }

    manager.apply_replacements(prompt)

    assert prompt["1"]["class_type"] == "NewNode"
    assert "_meta" not in prompt["1"]

def test_apply_replacements_fallback_to_id(manager):
    # This node doesn't have a display name mapping
    nodes.NODE_CLASS_MAPPINGS["NoDisplayNameNode"] = type("NoDisplayNameNode", (), {})

    manager.register(NodeReplace(old_node_id="OldNode", new_node_id="NoDisplayNameNode"))

    prompt = {
        "1": {
            "class_type": "OldNode",
            "inputs": {"a": 1},
            "_meta": {"title": "Old Node", "display_name": "Old Node"}
        }
    }

    manager.apply_replacements(prompt)

    assert prompt["1"]["class_type"] == "NoDisplayNameNode"
    assert prompt["1"]["_meta"].get("display_name") == "NoDisplayNameNode"
