# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Any, get_origin

import torch
import torch.utils._pytree as pytree
from typing_extensions import _AnnotatedAlias

from funsor import Funsor


def normalize_domain(domain: Any) -> _AnnotatedAlias:
    if get_origin(domain) is Annotated:
        assert domain.__origin__ is torch.Tensor
        metadata = domain.__metadata__
        if len(metadata) == 1:
            dtype = metadata[0]
            shape = ()
        elif len(metadata) == 2:
            dtype, shape = metadata
        else:
            raise ValueError(f"Invalid metadata: {metadata}. Expected (dtype,) or (dtype, shape).")

    elif isinstance(domain, torch.dtype) or domain in (int, float):
        dtype = domain
        shape = ()
    meta_val = torch.empty(shape, dtype=dtype, device="meta")
    return Annotated[type(meta_val), meta_val.dtype, meta_val.shape]


def check_funsor(x: Funsor, inputs: dict[str, Any], output: Any | None = None, data: Funsor | None = None) -> None:
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    inputs = pytree.tree_map(normalize_domain, inputs)
    assert x.inputs == inputs
    if output is not None:
        output = normalize_domain(output)
        assert x.output == output
