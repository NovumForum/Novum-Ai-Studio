import ast
import torch
import pytest
from comfy_extras.nodes_compositing import JoinImageWithAlpha, resize_mask

def old_join_image_with_alpha_execute(image: torch.Tensor, alpha: torch.Tensor):
    batch_size = min(len(image), len(alpha))
    out_images = []
    alpha_resized = 1.0 - resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:, :, :3], alpha_resized[i].unsqueeze(2)), dim=2))
    return torch.stack(out_images)

def test_join_image_with_alpha_execute_single_batch():
    image = torch.rand(1, 64, 64, 3)
    alpha = torch.rand(1, 64, 64)

    expected = old_join_image_with_alpha_execute(image, alpha)
    result = JoinImageWithAlpha.execute(image, alpha).args[0]

    assert result.shape == (1, 64, 64, 4)
    assert torch.allclose(result, expected, atol=1e-6)

def test_join_image_with_alpha_execute_multi_batch():
    image = torch.rand(4, 128, 128, 3)
    alpha = torch.rand(4, 128, 128)

    expected = old_join_image_with_alpha_execute(image, alpha)
    result = JoinImageWithAlpha.execute(image, alpha).args[0]

    assert result.shape == (4, 128, 128, 4)
    assert torch.allclose(result, expected, atol=1e-6)

def test_join_image_with_alpha_execute_mismatched_batch_size():
    image = torch.rand(5, 32, 32, 3)
    alpha = torch.rand(3, 32, 32)

    expected = old_join_image_with_alpha_execute(image, alpha)
    result = JoinImageWithAlpha.execute(image, alpha).args[0]

    assert result.shape == (3, 32, 32, 4)
    assert torch.allclose(result, expected, atol=1e-6)

def test_join_image_with_alpha_execute_rgba_input():
    image = torch.rand(2, 64, 64, 4)
    alpha = torch.rand(2, 64, 64)

    expected = old_join_image_with_alpha_execute(image, alpha)
    result = JoinImageWithAlpha.execute(image, alpha).args[0]

    assert result.shape == (2, 64, 64, 4)
    assert torch.allclose(result, expected, atol=1e-6)

def test_join_image_with_alpha_execute_mask_resizing():
    image = torch.rand(2, 64, 64, 3)
    alpha = torch.rand(2, 32, 32)

    expected = old_join_image_with_alpha_execute(image, alpha)
    result = JoinImageWithAlpha.execute(image, alpha).args[0]

    assert result.shape == (2, 64, 64, 4)
    assert torch.allclose(result, expected, atol=1e-6)

def test_join_image_with_alpha_no_python_for_loop():
    with open("comfy_extras/nodes_compositing.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "JoinImageWithAlpha":
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.FunctionDef) and subnode.name == "execute":
                    for_loops = [n for n in ast.walk(subnode) if isinstance(n, (ast.For, ast.While))]
                    assert len(for_loops) == 0, "JoinImageWithAlpha.execute should not contain loops"

def test_compositing_schema_tooltips():
    with open("comfy_extras/nodes_compositing.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    expected_classes = ["PorterDuffImageComposite", "SplitImageWithAlpha", "JoinImageWithAlpha"]
    found_classes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in expected_classes:
            found_classes.add(node.name)
            define_schema = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "define_schema")
            schema_call = next(n for n in ast.walk(define_schema) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Schema")

            keywords = {kw.arg: kw.value for kw in schema_call.keywords}
            assert "description" in keywords, f"Class {node.name} missing description in schema"
            assert "search_aliases" in keywords, f"Class {node.name} missing search_aliases in schema"

    assert found_classes == set(expected_classes)
