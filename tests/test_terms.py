# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import operator

import pytest
import torch

from funsor import ShapedTensor, Variable
from funsor.utilities.testing import (
    OPERATOR_BINARY_OPS,
    OPERATOR_BOOLEAN_OPS,
    TORCH_BINARY_OPS,
    check_funsor,
)


def test_shaped_tensor():
    assert ShapedTensor[torch.float32, (3, 4)] == ShapedTensor[torch.float32, (3, 4)]
    assert ShapedTensor[torch.float32, (3, 4)] != ShapedTensor[torch.float32, (3, 5)]
    assert ShapedTensor[torch.float32, (3, 4)] != ShapedTensor[torch.float32, ()]
    assert ShapedTensor[torch.float32, (3, 4)] != ShapedTensor[torch.float64, (3, 4)]


def test_cons_hash():
    assert Variable("x", int) is Variable("x", int)
    assert Variable("x", int) is Variable("x", int, ())
    assert Variable("x", int) is Variable("x", int, torch.Size([]))
    assert Variable("x", int) is not Variable("x", int, (1,))
    assert Variable("x", int) is not Variable("x", float)

    assert Variable("y", float) is Variable("y", float)
    assert Variable("y", float) is Variable("y", torch.float64)

    assert (Variable("x", int) + Variable("y", float)) is (Variable("x", int) + Variable("y", float))
    assert (Variable("x", int) + Variable("y", float)) is not (Variable("y", float) + Variable("x", int))


@pytest.mark.parametrize("dtype", [int, torch.float32])
def test_variable(dtype):
    x = Variable("x", dtype)
    check_funsor(x, {"x": dtype}, dtype)
    assert x("x") is x
    assert x(x) is x
    y = Variable("y", dtype)
    assert x("y") is y
    assert x(x="y") is y
    assert x(x=y) is y
    assert y is not x
    assert y(x) is x

    xp1 = x + 1.0
    assert xp1(x=2.0) == 3.0


def test_substitute():
    x = Variable("x", torch.float32)
    y = Variable("y", torch.float32)
    z = Variable("z", int)

    f = x * y + x * z
    check_funsor(f, {"x": torch.float32, "y": torch.float32, "z": int}, torch.float32)

    assert f(y=2) is x * 2 + x * z
    assert f(z=2) is x * y + x * 2
    assert f(y=x) is x * x + x * z
    assert f(x=y) is y * y + y * z
    assert f(y=z, z=y) is x * z + x * y
    assert f(x=y, y=z, z=x) is y * z + y * x


@pytest.mark.parametrize("op", ["abs", "neg"])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_operator_unary(op, value):
    expected = getattr(operator, op)(value)

    x = Variable("x", torch.float32)
    actual = getattr(operator, op)(x)(value)

    assert actual == expected


@pytest.mark.skip(reason="skipping math unary test")
@pytest.mark.parametrize("op", ["fabs", "ceil", "floor", "exp", "expm1", "log", "log1p", "sqrt"])
@pytest.mark.parametrize("value", [0.5, 1.0])
def test_math_unary(op, value):
    expected = getattr(math, op)(value)

    x = Variable("x", torch.float32)
    actual = getattr(math, op)(x)(value)

    assert actual == expected


@pytest.mark.parametrize("op", ["abs", "ceil", "floor", "exp", "expm1", "log", "log1p", "sqrt", "acos", "cos"])
@pytest.mark.parametrize("value", [torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)])
def test_torch_unary(op, value):
    expected = getattr(torch, op)(value)

    x = Variable("x", torch.float32)
    actual = getattr(torch, op)(x)(value)

    assert actual == expected


@pytest.mark.parametrize("value1", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("value2", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("binary_op", OPERATOR_BINARY_OPS + OPERATOR_BOOLEAN_OPS)
def test_operator_binary(binary_op, value1: float, value2: float):
    if binary_op in OPERATOR_BOOLEAN_OPS:
        value1 = bool(value1)
        value2 = bool(value2)
    try:
        expected = binary_op(value1, value2)
    except ZeroDivisionError:
        return

    x1 = Variable("x1", torch.float32)
    x2 = Variable("x2", torch.float32)
    actual = binary_op(x1, x2)(value1, value2)

    assert actual == expected


@pytest.mark.parametrize("value1", [torch.tensor(0.0), torch.tensor(0.2), torch.tensor(1.0)])
@pytest.mark.parametrize("value2", [torch.tensor(0.5), torch.tensor(1.0)])
@pytest.mark.parametrize("binary_op", TORCH_BINARY_OPS)
def test_torch_binary(binary_op, value1: torch.Tensor, value2: torch.Tensor):
    expected = binary_op(value1, value2)

    x1 = Variable("x1", value1.dtype)
    x2 = Variable("x2", value2.dtype)
    actual = binary_op(x1, x2)(value1, value2)

    assert actual == expected
