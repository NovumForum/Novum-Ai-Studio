import torch
import kornia
import comfy.cli_args

# Force CPU mode for unit testing environment without CUDA
comfy.cli_args.args.cpu = True

from unittest.mock import patch, MagicMock

# Mock nodes and server modules to prevent initialization side-effects during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_morphology import ImageRGBToYUV, ImageYUVToRGB


class TestYUVNodes:

    def test_rgb_to_yuv_execution(self):
        rgb_image = torch.rand(2, 64, 64, 3)
        res = ImageRGBToYUV.execute(rgb_image)

        assert len(res.result) == 3
        Y, U, V = res.result
        assert Y.shape == (2, 64, 64, 3)
        assert U.shape == (2, 64, 64, 3)
        assert V.shape == (2, 64, 64, 3)

        # Verify that channels within each returned tensor are duplicated
        assert torch.allclose(Y[..., 0], Y[..., 1])
        assert torch.allclose(Y[..., 0], Y[..., 2])

    def test_yuv_to_rgb_execution_3channel(self):
        rgb_image = torch.rand(2, 64, 64, 3)
        yuv_res = ImageRGBToYUV.execute(rgb_image)
        Y, U, V = yuv_res.result

        rgb_res = ImageYUVToRGB.execute(Y, U, V)
        reconstructed = rgb_res.result[0]

        assert reconstructed.shape == (2, 64, 64, 3)
        assert torch.allclose(rgb_image, reconstructed, atol=1e-3)

    def test_yuv_to_rgb_execution_1channel(self):
        Y = torch.rand(2, 32, 32, 1)
        U = torch.rand(2, 32, 32, 1)
        V = torch.rand(2, 32, 32, 1)

        res = ImageYUVToRGB.execute(Y, U, V)
        out = res.result[0]

        assert out.shape == (2, 32, 32, 3)

    def test_yuv_to_rgb_execution_3d_tensor(self):
        Y = torch.rand(2, 32, 32)
        U = torch.rand(2, 32, 32)
        V = torch.rand(2, 32, 32)

        res = ImageYUVToRGB.execute(Y, U, V)
        out = res.result[0]

        assert out.shape == (2, 32, 32, 3)

    def test_yuv_to_rgb_exact_equivalence_to_reference(self):
        # Generate test Y, U, V tensors with identical channels
        y_single = torch.rand(2, 64, 64, 1)
        u_single = torch.rand(2, 64, 64, 1)
        v_single = torch.rand(2, 64, 64, 1)

        Y = y_single.expand(2, 64, 64, 3)
        U = u_single.expand(2, 64, 64, 3)
        V = v_single.expand(2, 64, 64, 3)

        # Reference implementation
        ref_image = torch.cat([torch.mean(Y, dim=-1, keepdim=True), torch.mean(U, dim=-1, keepdim=True), torch.mean(V, dim=-1, keepdim=True)], dim=-1)
        ref_out = kornia.color.ycbcr_to_rgb(ref_image.movedim(-1, 1)).movedim(1, -1)

        # Optimized implementation
        opt_out = ImageYUVToRGB.execute(Y, U, V).result[0]

        assert torch.allclose(ref_out, opt_out, atol=1e-6)
