# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Annotated, Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.experimental.meta_tracer import (
    MetaProxy,
    MetaTracer,
    gen_constructor_wrapper,
    manual_meta_overrides,
)
from torch.fx.node import Argument, Target
from typing_extensions import _AnnotatedAlias


def node_to_meta(v):
    if isinstance(v, fx.Node):
        meta_val = v.meta.get("val")
        assert meta_val is not None, f"Node {v} does not have a meta value"
        return meta_val
    return v


class FunsorTracer(MetaTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.root = torch.nn.Module()
        self.graph = fx.Graph(tracer_cls=type(self))
        self.tensor_attrs: dict[torch.Tensor | ScriptObject | FakeScriptObject, str] = {}
        self.meta_args: dict[str, torch.Tensor] = {}
        self.funsor_cache: dict[Any, Funsor] = {}

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

    def __enter__(self) -> "FunsorTracer":
        tracer_stack.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        tracer_stack.pop()

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, torch.Tensor):
            meta_val = a.to(device="meta")
            a = super().create_arg(a)
            a.meta["val"] = meta_val
            return a
        return super().create_arg(a)

    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[fx.Node], fx.Proxy] | None = None,
    ) -> "Funsor":
        return Funsor(kind, target, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> fx.Node:
        # shape and dtype inference
        if kind == "placeholder":
            assert isinstance(target, str)
            meta_out = self.meta_args[target]
        else:
            if target in self.orig_fns:
                # NOTE: tensor constructors in PyTorch define the `device` argument as
                # *kwargs-only*. That is why this works. If you add methods to
                # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
                # this will break and you will likely see issues where we cannot infer
                # the size of the output.
                if "device" in kwargs:
                    kwargs["device"] = "meta"

            args_metas = pytree.tree_map(node_to_meta, args)
            kwargs_metas = pytree.tree_map(node_to_meta, kwargs)

            if kind == "call_function":
                assert callable(target)
                meta_target = manual_meta_overrides.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                assert isinstance(target, str)
                if target == "__getitem__":
                    # Scalar tensors lead to the following error:
                    # RuntimeError: Tensor.item() cannot be called on meta tensors
                    # This is a workaround to convert scalar tensors to Python scalars
                    indices = args_metas[1]
                    new_indices: tuple[int | float | torch.Tensor, ...] = ()
                    for index in indices:
                        if isinstance(index, torch.Tensor) and index.ndim == 0:
                            new_indices += (torch.tensor(0, dtype=index.dtype).item(),)
                        else:
                            new_indices += (index,)
                    args_metas = (args_metas[0], new_indices)
                    meta_out = getattr(args_metas[0], target)(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
                elif target == "__call__":
                    meta_out = args_metas[0]
                else:
                    meta_out = getattr(args_metas[0], target)(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
            elif kind == "call_module":
                assert isinstance(target, str)
                assert hasattr(self, "orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)  # type: ignore[misc, arg-type]
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                assert isinstance(target, str)
                self._disable_module_getattr = True
                try:
                    attr_itr: torch.nn.Module | torch.Tensor = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    assert isinstance(attr_itr, torch.Tensor)
                    meta_out = attr_itr.to(device="meta")
                finally:
                    self._disable_module_getattr = False
            else:
                raise ValueError(f"Unsupported kind {kind}")

        # Otherwise, create the node and cache it
        node = super().create_node(kind, target, args, kwargs)
        node.meta["val"] = meta_out
        return node


tracer_stack: list[FunsorTracer] = []
tracer_stack.append(FunsorTracer())


def is_impure_node(node: fx.Node) -> bool:
    if node.op == "placeholder":
        return False
    return node.is_impure()


class FunsorMeta(type):
    def __call__(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "Funsor":
        from funsor.interpretations import interpretation_stack

        self = interpretation_stack[-1].interpret(cls, *args, **kwargs)
        assert self is not None, f"Interpretation failed to construct {cls} with args {args} and kwargs {kwargs}"
        return self


class Funsor(MetaProxy, metaclass=FunsorMeta):
    """
    Abstract base class for immutable functional tensors.
    """

    tracer: FunsorTracer

    def __init__(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        tracer = tracer_stack[-1]
        args_ = tracer.create_arg(args)
        kwargs_ = tracer.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)
        node = tracer.create_node(kind, target, args_, kwargs_)
        self.tracer = tracer
        self.node = node
        self.install_tensor_meta(self.node.meta["val"])
        self._inputs: dict[str, _AnnotatedAlias] | None = None
        self._output: _AnnotatedAlias | None = None
        self._graph: fx.Graph | None = None

    @classmethod
    def make_hash_key(cls, kind: str, target: Target, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
        return (kind, target, args, tuple(kwargs.items()))

    @property
    def graph(self) -> fx.Graph:
        if self._graph is None:
            memo: dict[fx.Node, fx.Node] = {}
            self._graph = fx.Graph(tracer_cls=type(self.tracer))
            self._graph.graph_copy(self.tracer.graph, val_map=memo, return_output_node=False)
            self._graph.output(memo[self.node], type_expr=self.node.type)
            self._graph.eliminate_dead_code(is_impure_node=is_impure_node)
        return self._graph

    @property
    def inputs(self) -> dict[str, _AnnotatedAlias]:
        if self._inputs is None:
            self._inputs = {}
            for graph_node in self.graph.nodes:
                if graph_node.op == "placeholder":
                    meta_val = graph_node.meta["val"]
                    self._inputs[graph_node.name] = Annotated[torch.Tensor, meta_val.dtype, tuple(meta_val.shape)]
        return self._inputs

    @property
    def output(self) -> _AnnotatedAlias:
        if self._output is None:
            self._output = Annotated[torch.Tensor, self.dtype, tuple(self.shape)]
        return self._output

    def __repr__(self) -> str:
        return f"Funsor({', '.join([f'{name}: {domain}' for name, domain in self.inputs.items()])}) -> {self.output}"


class VariableMeta(FunsorMeta):
    def __call__(
        cls, name: str, dtype: type[int] | type[float] | torch.dtype, shape: tuple[int, ...] = ()
    ) -> "Variable":
        tracer = tracer_stack[-1]
        meta_val = torch.empty(shape, dtype=dtype, device="meta")  # type: ignore[arg-type]
        if name in tracer.meta_args:
            if tracer.meta_args[name].dtype != meta_val.dtype:
                raise ValueError(f"Expected dtype {tracer.meta_args[name].dtype} but got {meta_val.dtype}")
            if tracer.meta_args[name].shape != meta_val.shape:
                raise ValueError(f"Expected shape {tracer.meta_args[name].shape} but got {meta_val.shape}")
        else:
            tracer.meta_args[name] = meta_val
        self = super().__call__(name, dtype, shape)
        assert isinstance(self, Variable)
        return self


class Variable(Funsor, metaclass=VariableMeta):
    """
    Funsor representing a single free variable.

    Example:

    >>> x = Variable("x", int)
    >>> y = Variable("y", float)
    >>> f = x * y + x
    >>> f(x=1, y=2)

    Args:
        name:
            The name of the variable.
        dtype:
            The data type of the variable.
        shape:
            The shape of the variable.
    """

    def __init__(self, name: str, dtype: type[int] | type[float] | torch.dtype, shape: tuple[int, ...] = ()) -> None:
        super().__init__("placeholder", name, (), {})

    @classmethod
    def make_hash_key(  # type: ignore[override]
        cls, name: str, dtype: type[int] | type[float] | torch.dtype, shape: tuple[int, ...] = ()
    ) -> tuple:
        return ("placeholder", name, (), ())

    @property
    def name(self) -> str:
        return self.node.name

    def __repr__(self) -> str:
        return f"Variable({self.name}: {self.output})"
