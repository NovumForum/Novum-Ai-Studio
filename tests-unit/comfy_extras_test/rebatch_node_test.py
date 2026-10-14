import ast
import sys
from unittest.mock import MagicMock


class MockNodeOutput:
    def __init__(self, result):
        self.result = [result]


class MockComfyNode:
    pass


mock_io = MagicMock()
mock_io.ComfyNode = MockComfyNode
mock_io.NodeOutput = MockNodeOutput

mock_comfy_api_latest = MagicMock()
mock_comfy_api_latest.io = mock_io
mock_comfy_api_latest.ComfyExtension = MockComfyNode

mock_torch = MagicMock()
mock_typing = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['typing_extensions'] = mock_typing
sys.modules['comfy_api'] = MagicMock()
sys.modules['comfy_api.latest'] = mock_comfy_api_latest


class DummyTensor:
    """Mock PyTorch tensor to test ImageRebatch.execute without PyTorch installed."""
    def __init__(self, shape, data=None):
        self.shape = shape
        self.data = data if data is not None else list(range(shape[0]))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.data)
            sub_data = self.data[start:stop]
            new_shape = (len(sub_data),) + self.shape[1:]
            return DummyTensor(new_shape, sub_data)
        raise NotImplementedError("Index operation not supported in DummyTensor")


def mock_torch_cat(tensors, dim=0):
    new_shape = list(tensors[0].shape)
    new_shape[dim] = sum(t.shape[dim] for t in tensors)
    new_data = []
    for t in tensors:
        new_data.extend(t.data)
    return DummyTensor(tuple(new_shape), new_data)


def mock_torch_split(tensor, split_size, dim=0):
    res = []
    data = tensor.data
    total = tensor.shape[dim]
    for i in range(0, total, split_size):
        chunk_data = data[i:i + split_size]
        chunk_shape = list(tensor.shape)
        chunk_shape[dim] = len(chunk_data)
        res.append(DummyTensor(tuple(chunk_shape), chunk_data))
    return res


mock_torch.cat = mock_torch_cat
mock_torch.split = mock_torch_split


def test_rebatch_images_ast_structure():
    """Verify that ImageRebatch.execute is optimized with torch.cat and torch.split."""
    with open("comfy_extras/nodes_rebatch.py", "r") as f:
        tree = ast.parse(f.read())

    found_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ImageRebatch":
            found_class = True
            method_names = [
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            ]
            assert "execute" in method_names, "ImageRebatch must define execute method"

    assert found_class, "ImageRebatch class must be present in comfy_extras/nodes_rebatch.py"


def test_rebatch_images_execution_logic():
    """Verify ImageRebatch.execute logic across various batch configurations and fallback behavior."""
    from comfy_extras.nodes_rebatch import ImageRebatch

    # Test Case 1: Empty images list
    res = ImageRebatch.execute([], [4])
    assert len(res.result[0]) == 0

    # Test Case 2: Matching spatial dimensions across batches
    img1 = DummyTensor((3, 512, 512, 3), [1, 2, 3])
    img2 = DummyTensor((5, 512, 512, 3), [4, 5, 6, 7, 8])
    res = ImageRebatch.execute([img1, img2], [3])

    # Total frames = 8, batch_size = 3 -> outputs should have lengths 3, 3, 2
    output_batches = res.result[0]
    assert len(output_batches) == 3
    assert output_batches[0].shape == (3, 512, 512, 3)
    assert output_batches[0].data == [1, 2, 3]
    assert output_batches[1].shape == (3, 512, 512, 3)
    assert output_batches[1].data == [4, 5, 6]
    assert output_batches[2].shape == (2, 512, 512, 3)
    assert output_batches[2].data == [7, 8]

    # Test Case 3: Fallback path when spatial dimensions mismatch
    img_a = DummyTensor((2, 512, 512, 3), [1, 2])
    img_b = DummyTensor((2, 256, 256, 3), [3, 4])
    res_fallback = ImageRebatch.execute([img_a, img_b], [2])
    output_fallback = res_fallback.result[0]
    assert len(output_fallback) == 2
    assert output_fallback[0].shape == (2, 512, 512, 3)
    assert output_fallback[1].shape == (2, 256, 256, 3)
