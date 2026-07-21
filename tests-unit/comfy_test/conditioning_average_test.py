import torch
from nodes import ConditioningAverage


def test_conditioning_average():
    # Setup test conditioning vectors
    t0 = torch.rand((1, 5, 10))
    t1 = torch.rand((1, 5, 10))

    conditioning_to = [[t1, {"pooled_output": torch.rand((1, 15))}]]
    conditioning_from = [[t0, {"pooled_output": torch.rand((1, 15))}]]

    strength = 0.4

    # Run ConditioningAverage
    node = ConditioningAverage()
    out, = node.addWeighted(conditioning_to, conditioning_from, strength)

    # Expected results
    expected_tw = t0 + strength * (t1 - t0)
    expected_pooled = conditioning_from[0][1]["pooled_output"] + strength * (
        conditioning_to[0][1]["pooled_output"]
        - conditioning_from[0][1]["pooled_output"]
    )

    # Assertions
    assert torch.allclose(out[0][0], expected_tw, atol=1e-6)
    assert torch.allclose(out[0][1]["pooled_output"], expected_pooled, atol=1e-6)


def test_conditioning_average_padding():
    # Setup test with different shapes (t0 smaller)
    t0 = torch.rand((1, 3, 10))
    t1 = torch.rand((1, 5, 10))

    conditioning_to = [[t1, {"pooled_output": torch.rand((1, 15))}]]
    conditioning_from = [[t0, {"pooled_output": torch.rand((1, 15))}]]

    strength = 0.7

    node = ConditioningAverage()
    out, = node.addWeighted(conditioning_to, conditioning_from, strength)

    # t0 should be padded with zeros to match t1.shape[1]
    padded_t0 = torch.cat([t0, torch.zeros((1, 2, 10))], dim=1)
    expected_tw = padded_t0 + strength * (t1 - padded_t0)

    assert torch.allclose(out[0][0], expected_tw, atol=1e-6)
