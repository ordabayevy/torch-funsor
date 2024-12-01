# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from funsor import FunsorTracer, Variable


def test_cons_hash():
    with FunsorTracer():
        assert Variable("x", int) is Variable("x", int)
        with pytest.raises(ValueError, match="Expected dtype"):
            assert Variable("x", float) is not Variable("x", int)
        assert Variable("y", float) is Variable("y", float)
        assert Variable("y", float) is Variable("y", torch.float64)
        assert (Variable("x", int) + Variable("y", float)) is (Variable("x", int) + Variable("y", float))
        assert (Variable("x", int) + Variable("y", float)) is not (Variable("y", float) + Variable("x", int))
