import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_rgba():
    # Batch of 4 RGBA images (4 channels)
    image = torch.rand((4, 64, 64, 4))
    node_output = SplitImageWithAlpha.execute(image)
    out_images, out_masks = node_output.args[0], node_output.args[1]

    assert out_images.shape == (4, 64, 64, 3)
    assert out_masks.shape == (4, 64, 64)
    assert torch.allclose(out_images, image[..., :3])
    assert torch.allclose(out_masks, 1.0 - image[..., 3])


def test_split_image_with_alpha_rgb():
    # Batch of 4 RGB images (3 channels)
    image = torch.rand((4, 64, 64, 3))
    node_output = SplitImageWithAlpha.execute(image)
    out_images, out_masks = node_output.args[0], node_output.args[1]

    assert out_images.shape == (4, 64, 64, 3)
    assert out_masks.shape == (4, 64, 64)
    assert torch.allclose(out_images, image)
    assert torch.all(out_masks == 0.0)


def test_join_image_with_alpha_matching_size():
    image = torch.rand((4, 64, 64, 3))
    mask = torch.rand((4, 64, 64))

    node_output = JoinImageWithAlpha.execute(image, mask)
    out_image = node_output.args[0]

    assert out_image.shape == (4, 64, 64, 4)
    assert torch.allclose(out_image[..., :3], image)
    assert torch.allclose(out_image[..., 3], 1.0 - mask)


def test_join_image_with_alpha_mismatched_batch_and_mask_size():
    # Mismatched batch size and mask spatial size
    image = torch.rand((6, 64, 64, 3))
    mask = torch.rand((4, 32, 32))

    node_output = JoinImageWithAlpha.execute(image, mask)
    out_image = node_output.args[0]

    assert out_image.shape == (4, 64, 64, 4)
    assert torch.allclose(out_image[..., :3], image[:4])
