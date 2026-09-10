import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha, JoinImageWithAlpha


def test_split_image_with_alpha_4channel():
    image = torch.rand(4, 64, 64, 4)
    out = SplitImageWithAlpha.execute(image)
    out_images, out_alphas = out.args

    assert out_images.shape == (4, 64, 64, 3)
    assert out_alphas.shape == (4, 64, 64)
    assert torch.allclose(out_images, image[..., :3])
    assert torch.allclose(out_alphas, 1.0 - image[..., 3])


def test_split_image_with_alpha_3channel():
    image = torch.rand(4, 64, 64, 3)
    out = SplitImageWithAlpha.execute(image)
    out_images, out_alphas = out.args

    assert out_images.shape == (4, 64, 64, 3)
    assert out_alphas.shape == (4, 64, 64)
    assert torch.allclose(out_images, image)
    assert torch.all(out_alphas == 0.0)


def test_join_image_with_alpha_matching_shapes():
    image = torch.rand(4, 64, 64, 3)
    alpha = torch.rand(4, 64, 64)
    out = JoinImageWithAlpha.execute(image, alpha)
    out_image = out.args[0]

    assert out_image.shape == (4, 64, 64, 4)
    assert torch.allclose(out_image[..., :3], image)
    assert torch.allclose(out_image[..., 3], 1.0 - alpha)


def test_join_image_with_alpha_resized_mask():
    image = torch.rand(4, 64, 64, 3)
    alpha_small = torch.rand(4, 32, 32)
    out = JoinImageWithAlpha.execute(image, alpha_small)
    out_image = out.args[0]

    assert out_image.shape == (4, 64, 64, 4)
    assert torch.allclose(out_image[..., :3], image)


def test_join_image_with_alpha_batch_mismatch():
    image = torch.rand(4, 64, 64, 3)
    alpha = torch.rand(2, 64, 64)
    out = JoinImageWithAlpha.execute(image, alpha)
    out_image = out.args[0]

    assert out_image.shape == (2, 64, 64, 4)
    assert torch.allclose(out_image[..., :3], image[:2])
