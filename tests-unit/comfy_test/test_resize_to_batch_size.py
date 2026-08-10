import math
import torch
import pytest
from comfy.utils import resize_to_batch_size, resize_list_to_batch_size

# Original implementation for validation reference
def resize_to_batch_size_orig(tensor, batch_size):
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

def resize_list_to_batch_size_orig(l, batch_size):
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


def test_resize_to_batch_size_same():
    tensor = torch.randn(5, 10, 10)
    result = resize_to_batch_size(tensor, 5)
    assert result is tensor  # should return the same object when shape is unchanged


def test_resize_to_batch_size_one_or_less():
    tensor = torch.randn(5, 10, 10)
    result_0 = resize_to_batch_size(tensor, 0)
    assert result_0.shape[0] == 0

    result_1 = resize_to_batch_size(tensor, 1)
    assert result_1.shape[0] == 1
    assert torch.equal(result_1, tensor[:1])


@pytest.mark.parametrize("in_size, out_size", [
    (10, 5),    # Downscaling
    (10, 25),   # Upscaling
    (3, 8),     # Upscaling
    (15, 2),    # Downscaling
    (1, 10),    # Upscaling from single batch
])
def test_resize_to_batch_size_equivalence(in_size, out_size):
    tensor = torch.randn(in_size, 4, 4)
    result_opt = resize_to_batch_size(tensor, out_size)
    result_orig = resize_to_batch_size_orig(tensor, out_size)

    assert result_opt.shape[0] == out_size
    assert torch.equal(result_opt, result_orig)


def test_resize_list_to_batch_size_same():
    lst = [1, 2, 3, 4, 5]
    result = resize_list_to_batch_size(lst, 5)
    assert result == lst


def test_resize_list_to_batch_size_one_or_less():
    lst = [1, 2, 3, 4, 5]
    assert resize_list_to_batch_size(lst, 0) == []
    assert resize_list_to_batch_size(lst, 1) == [1]


@pytest.mark.parametrize("in_size, out_size", [
    (10, 5),
    (10, 25),
    (3, 8),
    (15, 2),
    (1, 10),
])
def test_resize_list_to_batch_size_equivalence(in_size, out_size):
    lst = list(range(in_size))
    result_opt = resize_list_to_batch_size(lst, out_size)
    result_orig = resize_list_to_batch_size_orig(lst, out_size)

    assert len(result_opt) == out_size
    assert result_opt == result_orig
