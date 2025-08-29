# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from typing import Any, cast, overload

import torch
from torch.utils._pytree import tree_map

from funsor.tensor import Tensor
from funsor.terms import Funsor, ShapedTensor

OPERATOR_COMPARISON_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.ge,
    operator.gt,
]

OPERATOR_BINARY_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
] + OPERATOR_COMPARISON_OPS

OPERATOR_BOOLEAN_OPS = [operator.and_, operator.or_, operator.xor]

TORCH_COMPARISON_OPS = [
    torch.eq,
    torch.ne,
    torch.lt,
    torch.le,
    torch.ge,
    torch.gt,
]

TORCH_BINARY_OPS = [
    torch.maximum,
    torch.minimum,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.pow,
    torch.logaddexp,
] + TORCH_COMPARISON_OPS

TORCH_BOOLEAN_OPS = [torch.logical_and, torch.logical_or, torch.logical_xor]


def normalize_domain(domain: int | float | torch.dtype | type[ShapedTensor]) -> type[ShapedTensor]:
    if isinstance(domain, torch.dtype) or domain is int or domain is float:
        return ShapedTensor[domain, ()]
    return cast(type[ShapedTensor], domain)


def tensor_to_meta(x: Any) -> torch.Tensor:
    if isinstance(x, (torch.Tensor, Tensor)):
        return torch.empty(x.shape, dtype=x.dtype, device="meta")
    return x


def eval_shape(func, *args, **kwargs) -> type[ShapedTensor]:
    args, kwargs = tree_map(tensor_to_meta, [args, kwargs])
    meta_val = func(*args, **kwargs)
    return ShapedTensor[meta_val.dtype, meta_val.shape]


def check_funsor(
    x: torch.Tensor | Funsor,
    inputs: dict[str, type[ShapedTensor]],
    output: type[ShapedTensor],
    data: torch.Tensor | None = None,
) -> None:
    """
    Check dims and shape modulo reordering.
    """
    if isinstance(x, torch.Tensor):
        assert inputs == {}
        output = normalize_domain(output)
        assert output.dtype == x.dtype
        assert output.shape == x.shape
        if data is not None:
            # Use torch.allclose for tensor comparison to handle multi-element tensors
            assert torch.allclose(data, x, rtol=1e-05, atol=1e-08, equal_nan=True)
    elif isinstance(x, Funsor):
        inputs = tree_map(normalize_domain, inputs)
        assert x.inputs == inputs
        output = normalize_domain(output)
        assert x.output == output
        if data is not None:
            assert isinstance(x, Tensor)
            assert torch.allclose(x.data, data, rtol=1e-05, atol=1e-08, equal_nan=True)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")


@overload
def assert_equiv(x: Tensor, y: Tensor) -> None: ...
@overload
def assert_equiv(x: torch.Tensor, y: torch.Tensor) -> None: ...
def assert_equiv(x: Tensor | torch.Tensor, y: Tensor | torch.Tensor) -> None:
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        assert torch.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=True)
    elif isinstance(x, Tensor) and isinstance(y, Tensor):
        check_funsor(x, y.inputs, y.output, y.align(x.indices).data)
    else:
        raise ValueError(f"Expected both arguments to be Tensors or torch.Tensors, got {type(x)} and {type(y)}")
