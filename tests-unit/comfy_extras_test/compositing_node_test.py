import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_4channels():
    # Shape [Batch=4, Height=64, Width=64, Channels=4]
    image = torch.rand((4, 64, 64, 4))
    node_output = SplitImageWithAlpha.execute(image)

    out_image, out_alpha = node_output.result

    assert out_image.shape == (4, 64, 64, 3)
    assert out_alpha.shape == (4, 64, 64)
    assert torch.equal(out_image, image[..., :3])
    assert torch.allclose(out_alpha, 1.0 - image[..., 3])


def test_split_image_with_alpha_3channels():
    # Shape [Batch=2, Height=32, Width=32, Channels=3]
    image = torch.rand((2, 32, 32, 3))
    node_output = SplitImageWithAlpha.execute(image)

    out_image, out_alpha = node_output.result

    assert out_image.shape == (2, 32, 32, 3)
    assert out_alpha.shape == (2, 32, 32)
    assert torch.equal(out_image, image)
    assert torch.allclose(out_alpha, torch.zeros_like(image[..., 0]))


def test_join_image_with_alpha_matching_batch():
    # Image shape [4, 64, 64, 3], alpha shape [4, 64, 64]
    image = torch.rand((4, 64, 64, 3))
    alpha = torch.rand((4, 64, 64))

    node_output = JoinImageWithAlpha.execute(image, alpha)
    (out_image,) = node_output.result

    assert out_image.shape == (4, 64, 64, 4)
    assert torch.equal(out_image[..., :3], image)


def test_join_image_with_alpha_mismatched_batch_and_resize():
    # Mismatched batch sizes and mask dimensions
    image = torch.rand((4, 64, 64, 3))
    alpha = torch.rand((2, 32, 32))

    node_output = JoinImageWithAlpha.execute(image, alpha)
    (out_image,) = node_output.result

    # Min batch size is 2
    assert out_image.shape == (2, 64, 64, 4)
    assert torch.equal(out_image[..., :3], image[:2])
