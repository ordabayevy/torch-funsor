# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from contextlib import ContextDecorator
from typing import Any

import torch
from torch.fx.node import Target

from funsor.tensor import Tensor, TensorMeta
from funsor.terms import Funsor, FunsorMeta, tracer_stack


class Interpretation(ContextDecorator):
    """
    Abstract base class for Funsor interpretations.
    """

    def __init__(self, name: str) -> None:
        self.__name__ = name
        super().__init__()

    def __repr__(self) -> str:
        return self.__name__

    def __enter__(self) -> "Interpretation":
        interpretation_stack.append(self)
        return self

    def __exit__(self, *exc) -> None:
        interpretation_stack.pop()

    def interpret(self, cls: FunsorMeta, *args: Any, **kwargs: Any) -> Funsor | None:
        """
        Interpret a Funsor class and its arguments.
        """
        raise NotImplementedError


class CallableInterpretation(Interpretation):
    """
    Interpretation that calls a function.
    """

    def __init__(self, interpret_fn: Callable) -> None:
        super().__init__(interpret_fn.__name__)
        self.interpret_fn = interpret_fn

    def interpret(self, cls: FunsorMeta, *args: Any, **kwargs: Any) -> Funsor:
        return self.interpret_fn(cls, *args, **kwargs)


class PatternMatchingInterpretation(Interpretation):
    """
    Interpretation that dispatches to a function based on a pattern.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pattern_handlers: list[Callable] = []

    def register(self, pattern_handler: Callable) -> Callable:
        """Register a new pattern with its handler."""
        self.pattern_handlers.append((pattern_handler))
        return pattern_handler

    def interpret(self, cls, *args, **kwargs) -> Funsor | None:
        """Perform structural pattern matching on the input."""
        kind, target, args, kwargs, type_expr = cls.make_hash_key(*args, **kwargs)  # type: ignore[attr-defined]
        pattern = (kind, target, args, dict(kwargs), type_expr)
        for pattern_handler in self.pattern_handlers:
            result = pattern_handler(cls, pattern)
            if result is not None:
                return result
        return None


class PrioritizedInterpretation(Interpretation):
    """
    Interpretation that delegates to a list of interpretations.
    """

    def __init__(self, *subinterpretations: Interpretation) -> None:
        super().__init__("/".join(s.__name__ for s in subinterpretations))
        self.subinterpretations = subinterpretations

    def interpret(self, cls, *args, **kwargs) -> Funsor | None:
        for interpretation in self.subinterpretations:
            result = interpretation.interpret(cls, *args, **kwargs)
            if result is not None:
                return result
        return None


@CallableInterpretation
def reflect(cls: FunsorMeta, *args: Any, **kwargs: Any) -> Funsor:
    """
    Construct a funsor and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    # Create a cache key based on the arguments
    cache_key = cls.make_hash_key(*args, **kwargs)  # type: ignore[attr-defined]
    tracer = tracer_stack[-1]

    # Check if the node already exists in the cache
    if cache_key in tracer.funsor_cache:
        return tracer.funsor_cache[cache_key]
    else:
        self = tracer.funsor_cache[cache_key] = super(FunsorMeta, cls).__call__(*args, **kwargs)
        return self


eager_base = PatternMatchingInterpretation("eager")
eager = PrioritizedInterpretation(eager_base, reflect)
"""
Eager exact naive interpretation wherever possible.
"""


@eager_base.register
def eager_tensor(
    cls: FunsorMeta | TensorMeta, pattern: tuple[str, Target, tuple[Any, ...], dict[str, Any], Any | None]
) -> Funsor | None:
    match pattern:
        case ("call_method", "__getitem__", (torch.Tensor() as data, indices), _, None):
            if cls is Tensor:
                return None
            return Tensor(data, indices)
    return None


@eager_base.register
def eager_reduce(
    cls: FunsorMeta | TensorMeta, pattern: tuple[str, Target, tuple[Any, ...], dict[str, Any], Any | None]
) -> Funsor | None:
    """
    Interpret a Funsor reduction call.
    """
    match pattern:
        case ("call_function", "reduce", (Funsor() as f, *args), kwargs, None):
            return f.reduce(*args, **kwargs)
    return None


interpretation_stack: list[Interpretation] = []
interpretation_stack.append(eager)
