import torch
from comfy_extras.nodes_compositing import SplitImageWithAlpha


def test_split_image_with_alpha_rgba():
    """Verify SplitImageWithAlpha.execute with 4-channel RGBA input tensors across various batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        # Generate random RGBA tensor (B, H, W, C)
        image_rgba = torch.rand((batch_size, 32, 32, 4), dtype=torch.float32)

        # Reference unvectorized implementation
        out_images_ref = [i[:, :, :3] for i in image_rgba]
        out_alphas_ref = [i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image_rgba]
        ref_img, ref_mask = torch.stack(out_images_ref), 1.0 - torch.stack(out_alphas_ref)

        # Optimized execution
        node_output = SplitImageWithAlpha.execute(image_rgba)
        opt_img, opt_mask = node_output[0], node_output[1]

        # Check shapes
        assert opt_img.shape == (batch_size, 32, 32, 3)
        assert opt_mask.shape == (batch_size, 32, 32)

        # Check exact value equivalence
        assert torch.equal(opt_img, ref_img)
        assert torch.equal(opt_mask, ref_mask)


def test_split_image_with_alpha_rgb():
    """Verify SplitImageWithAlpha.execute with 3-channel RGB input tensors (no alpha channel)."""
    for batch_size in [1, 2, 4]:
        image_rgb = torch.rand((batch_size, 16, 16, 3), dtype=torch.float32)

        # Reference unvectorized implementation
        out_images_ref = [i[:, :, :3] for i in image_rgb]
        out_alphas_ref = [i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image_rgb]
        ref_img, ref_mask = torch.stack(out_images_ref), 1.0 - torch.stack(out_alphas_ref)

        node_output = SplitImageWithAlpha.execute(image_rgb)
        opt_img, opt_mask = node_output[0], node_output[1]

        assert opt_img.shape == (batch_size, 16, 16, 3)
        assert opt_mask.shape == (batch_size, 16, 16)

        assert torch.equal(opt_img, ref_img)
        assert torch.equal(opt_mask, ref_mask)
        # All alpha mask values should be zero (inverted from ones)
        assert torch.all(opt_mask == 0.0)
