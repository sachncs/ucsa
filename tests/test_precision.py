import pytest
import torch
from torch import nn

from ucsa.utils import precision


def test_the_library_dtype_is_a_single_floating_type():
    assert precision.DTYPE.is_floating_point
    assert precision.DTYPE in (torch.float32, torch.float16)


def test_configure_makes_it_the_default_for_new_tensors():
    precision.configure()
    assert torch.zeros(2).dtype == precision.DTYPE
    assert nn.Linear(2, 2).weight.dtype == precision.DTYPE


def test_to_dtype_is_exact_for_masks_and_ids():
    mask = torch.tensor([True, False, True])
    out = precision.to_dtype(mask)
    assert out.dtype == precision.DTYPE
    assert out.tolist() == [1.0, 0.0, 1.0]
    assert precision.to_dtype(torch.tensor([3, 5])).tolist() == [3.0, 5.0]


def test_to_dtype_refuses_float_to_float_conversion():
    with pytest.raises(TypeError, match="integer and boolean"):
        precision.to_dtype(torch.zeros(2))


def test_adam_epsilon_is_representable_in_the_dtype():
    assert torch.tensor(precision.ADAM_EPS, dtype=precision.DTYPE) > 0
