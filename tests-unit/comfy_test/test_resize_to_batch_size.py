import torch
import math
from comfy.utils import resize_to_batch_size, resize_list_to_batch_size

def resize_to_batch_size_reference(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor
    if batch_size <= 1:
        return tensor[:batch_size]
    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]
    return output

def resize_list_to_batch_size_reference(l, batch_size):
    in_batch_size = len(l)
    if in_batch_size == batch_size or in_batch_size == 0:
        return l
    if batch_size <= 1:
        return l[:batch_size]
    output = []
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output.append(l[min(round(i * scale), in_batch_size - 1)])
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
           output.append(l[min(math.floor((i + 0.5) * scale), in_batch_size - 1)])
    return output

def test_resize_to_batch_size_identical_or_small():
    # Test identical batch size
    t = torch.randn(5, 3, 16, 16)
    out = resize_to_batch_size(t, 5)
    assert out is t

    # Test batch size <= 1
    out_1 = resize_to_batch_size(t, 1)
    assert out_1.shape == (1, 3, 16, 16)
    assert torch.all(out_1[0] == t[0])

    out_0 = resize_to_batch_size(t, 0)
    assert out_0.shape == (0, 3, 16, 16)

def test_resize_to_batch_size_equivalence():
    # Verify exact equivalence for upscaling & downscaling for many size combinations
    for in_size in [2, 3, 5, 10, 50, 120, 199]:
        for out_size in [2, 3, 5, 10, 50, 120, 199]:
            if in_size == out_size:
                continue
            t = torch.randn(in_size, 2, 4)
            ref = resize_to_batch_size_reference(t, out_size)
            opt = resize_to_batch_size(t, out_size)
            assert ref.shape == opt.shape
            assert torch.all(ref == opt)

def test_resize_to_batch_size_dtypes():
    # Ensure it works with float32, float16, int32, and keeps device
    for dtype in [torch.float32, torch.float16, torch.int32]:
        t = torch.arange(10, dtype=dtype).reshape(10, 1)
        out = resize_to_batch_size(t, 5)
        assert out.dtype == dtype
        assert out.device == t.device

def test_resize_list_to_batch_size_equivalence():
    for in_size in [0, 1, 2, 5, 20, 99]:
        for out_size in [0, 1, 2, 5, 20, 99]:
            l = list(range(in_size))
            ref = resize_list_to_batch_size_reference(l, out_size)
            opt = resize_list_to_batch_size(l, out_size)
            assert ref == opt
