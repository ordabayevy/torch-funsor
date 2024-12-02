# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from funsor import FunsorTracer, Variable
from funsor.utilities.testing import check_funsor


def test_cons_hash():
    with FunsorTracer():
        assert Variable("x", int) is Variable("x", int)
        with pytest.raises(ValueError, match="Expected dtype"):
            assert Variable("x", float) is not Variable("x", int)
        assert Variable("y", float) is Variable("y", float)
        assert Variable("y", float) is Variable("y", torch.float64)
        assert (Variable("x", int) + Variable("y", float)) is (Variable("x", int) + Variable("y", float))
        assert (Variable("x", int) + Variable("y", float)) is not (Variable("y", float) + Variable("x", int))


@pytest.mark.parametrize("dtype", [int, torch.float32])
def test_variable(dtype):
    with FunsorTracer():
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
    with FunsorTracer():
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
