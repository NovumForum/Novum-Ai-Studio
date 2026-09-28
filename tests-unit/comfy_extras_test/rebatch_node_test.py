import torch
from comfy_extras.nodes_rebatch import ImageRebatch


def test_image_rebatch_single_tensor():
    # Test rebatching a single tensor of batch size 10 into batches of size 3
    img = torch.randn(10, 32, 32, 3)
    out = ImageRebatch.execute([img], [3]).args[0]

    assert len(out) == 4
    assert out[0].shape == (3, 32, 32, 3)
    assert out[1].shape == (3, 32, 32, 3)
    assert out[2].shape == (3, 32, 32, 3)
    assert out[3].shape == (1, 32, 32, 3)

    # Verify zero-copy slicing / exact data equality
    assert torch.equal(torch.cat(out, dim=0), img)


def test_image_rebatch_single_tensor_smaller_than_batch_size():
    img = torch.randn(2, 32, 32, 3)
    out = ImageRebatch.execute([img], [5]).args[0]

    assert len(out) == 1
    assert out[0].shape == (2, 32, 32, 3)
    assert torch.equal(out[0], img)


def test_image_rebatch_multiple_tensors_exact():
    # 3 input tensors of batch size 4 each (total 12 images), target batch size 4
    img1 = torch.randn(4, 16, 16, 3)
    img2 = torch.randn(4, 16, 16, 3)
    img3 = torch.randn(4, 16, 16, 3)

    out = ImageRebatch.execute([img1, img2, img3], [4]).args[0]

    assert len(out) == 3
    assert out[0].shape == (4, 16, 16, 3)
    assert out[1].shape == (4, 16, 16, 3)
    assert out[2].shape == (4, 16, 16, 3)

    assert torch.equal(out[0], img1)
    assert torch.equal(out[1], img2)
    assert torch.equal(out[2], img3)


def test_image_rebatch_multiple_tensors_straddle():
    # 2 input tensors of sizes 3 and 5 (total 8), target batch size 4
    img1 = torch.randn(3, 16, 16, 3)
    img2 = torch.randn(5, 16, 16, 3)

    out = ImageRebatch.execute([img1, img2], [4]).args[0]

    assert len(out) == 2
    assert out[0].shape == (4, 16, 16, 3)
    assert out[1].shape == (4, 16, 16, 3)

    expected_all = torch.cat([img1, img2], dim=0)
    assert torch.equal(torch.cat(out, dim=0), expected_all)


def test_image_rebatch_empty_input():
    out = ImageRebatch.execute([], [4]).args[0]
    assert out == []
