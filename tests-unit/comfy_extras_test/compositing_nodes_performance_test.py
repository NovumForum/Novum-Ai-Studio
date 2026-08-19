import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_rgba():
    batch_size, height, width = 4, 64, 64
    image = torch.rand(batch_size, height, width, 4)

    output = SplitImageWithAlpha.execute(image)
    out_images, out_masks = output.args[0], output.args[1]

    assert out_images.shape == (batch_size, height, width, 3)
    assert out_masks.shape == (batch_size, height, width)

    torch.testing.assert_close(out_images, image[..., :3])
    torch.testing.assert_close(out_masks, 1.0 - image[..., 3])


def test_split_image_with_alpha_rgb():
    batch_size, height, width = 4, 64, 64
    image = torch.rand(batch_size, height, width, 3)

    output = SplitImageWithAlpha.execute(image)
    out_images, out_masks = output.args[0], output.args[1]

    assert out_images.shape == (batch_size, height, width, 3)
    assert out_masks.shape == (batch_size, height, width)

    torch.testing.assert_close(out_images, image)
    torch.testing.assert_close(out_masks, torch.zeros((batch_size, height, width)))


def test_join_image_with_alpha():
    batch_size, height, width = 4, 64, 64
    image = torch.rand(batch_size, height, width, 3)
    alpha_mask = torch.rand(batch_size, height, width)

    output = JoinImageWithAlpha.execute(image, alpha_mask)
    out_images = output.args[0]

    assert out_images.shape == (batch_size, height, width, 4)
    torch.testing.assert_close(out_images[..., :3], image)
    torch.testing.assert_close(out_images[..., 3], 1.0 - alpha_mask)


def test_join_image_with_alpha_mismatched_batch():
    image = torch.rand(2, 64, 64, 3)
    alpha_mask = torch.rand(4, 64, 64)

    output = JoinImageWithAlpha.execute(image, alpha_mask)
    out_images = output.args[0]

    assert out_images.shape == (2, 64, 64, 4)
    torch.testing.assert_close(out_images[..., :3], image)
    torch.testing.assert_close(out_images[..., 3], 1.0 - alpha_mask[:2])
