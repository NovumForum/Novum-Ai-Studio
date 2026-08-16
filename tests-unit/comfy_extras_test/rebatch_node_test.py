import pytest
import torch
from comfy_extras.nodes_rebatch import ImageRebatch, LatentRebatch


def test_image_rebatch_basic():
    # Test batching multiple images of varying initial batch sizes
    images = [
        torch.randn(3, 64, 64, 3),
        torch.randn(5, 64, 64, 3),
        torch.randn(2, 64, 64, 3),
    ] # Total 10 images
    batch_size = [4]

    output = ImageRebatch.execute(images, batch_size)
    result = output.args[0]

    # Should split 10 images into chunks of 4, 4, 2
    assert len(result) == 3
    assert result[0].shape == (4, 64, 64, 3)
    assert result[1].shape == (4, 64, 64, 3)
    assert result[2].shape == (2, 64, 64, 3)

    # Verify tensor contents match original order
    all_original = torch.cat(images, dim=0)
    all_rebatched = torch.cat(result, dim=0)
    assert torch.equal(all_original, all_rebatched)


def test_image_rebatch_single_batch():
    # When batch size matches input or is 1
    images = [torch.randn(1, 32, 32, 3) for _ in range(5)]
    batch_size = [1]

    output = ImageRebatch.execute(images, batch_size)
    result = output.args[0]

    assert len(result) == 5
    for img in result:
        assert img.shape == (1, 32, 32, 3)


def test_image_rebatch_larger_batch_size():
    # When target batch size is larger than total input images
    images = [torch.randn(2, 32, 32, 3), torch.randn(3, 32, 32, 3)]
    batch_size = [10]

    output = ImageRebatch.execute(images, batch_size)
    result = output.args[0]

    assert len(result) == 1
    assert result[0].shape == (5, 32, 32, 3)
