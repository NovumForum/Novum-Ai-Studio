import torch
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

mock_server = MagicMock()

with patch.dict('sys.modules', {'nodes': mock_nodes, 'server': mock_server}):
    from comfy_extras.nodes_mask import FeatherMask


class TestFeatherMask:

    def test_zero_feathering(self):
        """Test that zero feathering on all sides leaves the mask unchanged."""
        mask = torch.ones((1, 64, 64), dtype=torch.float32)
        res = FeatherMask.execute(mask, 0, 0, 0, 0)[0]
        assert torch.equal(res, mask)

    def test_left_feathering(self):
        """Test feathering on the left edge produces linear ramp from 1/left to 1.0."""
        mask = torch.ones((1, 32, 32), dtype=torch.float32)
        left = 8
        res = FeatherMask.execute(mask, left, 0, 0, 0)[0]

        # First pixel column should be 1/8
        assert torch.isclose(res[0, 0, 0], torch.tensor(1.0 / left))
        # 8th pixel column (index 7) should be 1.0
        assert torch.isclose(res[0, 0, left - 1], torch.tensor(1.0))
        # Non-feathered area should remain 1.0
        assert torch.isclose(res[0, 0, left], torch.tensor(1.0))

    def test_right_feathering(self):
        """Test feathering on the right edge produces linear ramp."""
        mask = torch.ones((1, 32, 32), dtype=torch.float32)
        right = 8
        res = FeatherMask.execute(mask, 0, 0, right, 0)[0]

        # Last pixel column (index -1) should be 1/right
        assert torch.isclose(res[0, 0, -1], torch.tensor(1.0 / right))
        # Boundary column (index -right) should be 1.0
        assert torch.isclose(res[0, 0, -right], torch.tensor(1.0))

    def test_top_feathering(self):
        """Test feathering on the top edge."""
        mask = torch.ones((1, 32, 32), dtype=torch.float32)
        top = 8
        res = FeatherMask.execute(mask, 0, top, 0, 0)[0]

        # First row should be 1/top
        assert torch.isclose(res[0, 0, 0], torch.tensor(1.0 / top))
        # Row index top-1 should be 1.0
        assert torch.isclose(res[0, top - 1, 0], torch.tensor(1.0))

    def test_bottom_feathering(self):
        """Test feathering on the bottom edge."""
        mask = torch.ones((1, 32, 32), dtype=torch.float32)
        bottom = 8
        res = FeatherMask.execute(mask, 0, 0, 0, bottom)[0]

        # Last row should be 1/bottom
        assert torch.isclose(res[0, -1, 0], torch.tensor(1.0 / bottom))
        # Row index -bottom should be 1.0
        assert torch.isclose(res[0, -bottom, 0], torch.tensor(1.0))

    def test_batch_processing(self):
        """Test feathering across multi-batch tensors."""
        mask = torch.ones((4, 32, 32), dtype=torch.float32)
        res = FeatherMask.execute(mask, 4, 4, 4, 4)[0]

        assert res.shape == (4, 32, 32)
        for b in range(4):
            assert torch.isclose(res[b, 0, 0], torch.tensor((1.0 / 4) * (1.0 / 4)))

    def test_edge_bounds_clamping(self):
        """Test when requested feathering exceeds image dimensions."""
        mask = torch.ones((1, 16, 16), dtype=torch.float32)
        res = FeatherMask.execute(mask, 100, 100, 100, 100)[0]
        assert res.shape == (1, 16, 16)
