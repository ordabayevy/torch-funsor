# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from contextlib import ContextDecorator
from typing import Any

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

    def interpret(self, cls: FunsorMeta, *args: Any, **kwargs: Any) -> Funsor:
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


interpretation_stack: list[Interpretation] = []
interpretation_stack.append(reflect)
