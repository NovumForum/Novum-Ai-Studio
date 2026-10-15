import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
from nodes import ImagePadForOutpaint


def reference_expand_image(image, left, top, right, bottom, feathering):
    d1, d2, d3, d4 = image.size()

    new_image = torch.ones(
        (d1, d2 + top + bottom, d3 + left + right, d4),
        dtype=torch.float32,
        device=image.device,
    ) * 0.5

    new_image[:, top:top + d2, left:left + d3, :] = image

    mask = torch.ones(
        (d2 + top + bottom, d3 + left + right),
        dtype=torch.float32,
        device=image.device,
    )

    t = torch.zeros(
        (d2, d3),
        dtype=torch.float32,
        device=image.device,
    )

    if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:
        for i in range(d2):
            for j in range(d3):
                dt = i if top != 0 else d2
                db = d2 - i if bottom != 0 else d2

                dl = j if left != 0 else d3
                dr = d3 - j if right != 0 else d3

                d = min(dt, db, dl, dr)

                if d >= feathering:
                    continue

                v = (feathering - d) / feathering

                t[i, j] = v * v

    mask[top:top + d2, left:left + d3] = t

    return (new_image, mask.unsqueeze(0))


def test_image_pad_outpaint_basic():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 100, 100, 3)

    out_image, out_mask = node.expand_image(image, left=10, top=20, right=10, bottom=20, feathering=10)
    ref_image, ref_mask = reference_expand_image(image, left=10, top=20, right=10, bottom=20, feathering=10)

    assert out_image.shape == (1, 140, 120, 3)
    assert out_mask.shape == (1, 140, 120)
    assert torch.allclose(out_image, ref_image, atol=1e-6)
    assert torch.allclose(out_mask, ref_mask, atol=1e-6)


def test_image_pad_outpaint_zero_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand(2, 64, 64, 3)

    out_image, out_mask = node.expand_image(image, left=8, top=8, right=8, bottom=8, feathering=0)
    ref_image, ref_mask = reference_expand_image(image, left=8, top=8, right=8, bottom=8, feathering=0)

    assert torch.allclose(out_image, ref_image, atol=1e-6)
    assert torch.allclose(out_mask, ref_mask, atol=1e-6)


def test_image_pad_outpaint_excessive_feathering():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 50, 50, 3)

    # Feathering * 2 >= d2/d3 => feathering check fails, no feathering applied
    out_image, out_mask = node.expand_image(image, left=10, top=10, right=10, bottom=10, feathering=30)
    ref_image, ref_mask = reference_expand_image(image, left=10, top=10, right=10, bottom=10, feathering=30)

    assert torch.allclose(out_image, ref_image, atol=1e-6)
    assert torch.allclose(out_mask, ref_mask, atol=1e-6)


def test_image_pad_outpaint_asymmetric_zero_padding():
    node = ImagePadForOutpaint()
    image = torch.rand(1, 80, 80, 3)

    out_image, out_mask = node.expand_image(image, left=0, top=10, right=20, bottom=0, feathering=15)
    ref_image, ref_mask = reference_expand_image(image, left=0, top=10, right=20, bottom=0, feathering=15)

    assert torch.allclose(out_image, ref_image, atol=1e-6)
    assert torch.allclose(out_mask, ref_mask, atol=1e-6)
