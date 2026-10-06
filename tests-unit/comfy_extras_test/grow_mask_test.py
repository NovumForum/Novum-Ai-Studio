import ast
import unittest
import torch
import scipy.ndimage
import numpy as np

# Set cpu mode flag before importing comfy_extras
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy_extras.nodes_mask import GrowMask


def reference_grow_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask_reshaped = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask_reshaped:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


class TestGrowMaskNode(unittest.TestCase):
    def test_schema_and_ast_structure(self):
        with open("comfy_extras/nodes_mask.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "GrowMask":
                found = True
                break
        self.assertTrue(found, "GrowMask class definition not found in comfy_extras/nodes_mask.py")

        schema = GrowMask.define_schema()
        self.assertEqual(schema.node_id, "GrowMask")
        self.assertEqual(schema.display_name, "Grow Mask")

    def test_grow_mask_zero_expansion(self):
        mask = torch.rand((2, 64, 64), dtype=torch.float32)
        res = GrowMask.execute(mask, expand=0, tapered_corners=True).output[0]
        self.assertEqual(res.shape, (2, 64, 64))
        torch.testing.assert_close(res, mask)

    def test_grow_mask_positive_expansion_tapered(self):
        mask = torch.zeros((1, 32, 32), dtype=torch.float32)
        mask[0, 16, 16] = 1.0
        res = GrowMask.execute(mask, expand=3, tapered_corners=True).output[0]
        ref = reference_grow_mask(mask, expand=3, tapered_corners=True)
        self.assertEqual(res.shape, ref.shape)
        torch.testing.assert_close(res, ref)

    def test_grow_mask_positive_expansion_square(self):
        mask = torch.zeros((2, 32, 32), dtype=torch.float32)
        mask[:, 10:15, 10:15] = 1.0
        res = GrowMask.execute(mask, expand=2, tapered_corners=False).output[0]
        ref = reference_grow_mask(mask, expand=2, tapered_corners=False)
        self.assertEqual(res.shape, ref.shape)
        torch.testing.assert_close(res, ref)

    def test_grow_mask_negative_expansion(self):
        mask = torch.ones((1, 32, 32), dtype=torch.float32)
        mask[0, 0:5, 0:5] = 0.0
        res = GrowMask.execute(mask, expand=-2, tapered_corners=True).output[0]
        ref = reference_grow_mask(mask, expand=-2, tapered_corners=True)
        self.assertEqual(res.shape, ref.shape)
        torch.testing.assert_close(res, ref)


if __name__ == "__main__":
    unittest.main()
