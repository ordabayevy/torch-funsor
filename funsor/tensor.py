# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause


import operator

import torch
import torch.fx as fx
from torch.fx.graph import magic_methods, reflectable_magic_methods
from torch.utils._pytree import tree_any, tree_leaves, tree_map

from funsor.terms import Funsor, FunsorMeta, Variable


def _expand_ellipsis(indices: tuple, tensor_shape: tuple) -> tuple:
    """Expand ellipsis in indexing arguments to match tensor dimensions."""
    # Find the position of the ellipsis by checking each argument individually
    ellipsis_pos = None
    for i, idx in enumerate(indices):
        if idx is Ellipsis:
            ellipsis_pos = i
            break

    # If no ellipsis found, return indices as-is
    if ellipsis_pos is None:
        return indices

    # Count non-ellipsis dimensions
    num_explicit_dims = len(indices) - 1  # subtract 1 for the ellipsis
    num_tensor_dims = len(tensor_shape)

    # Calculate how many dimensions the ellipsis should expand to
    num_ellipsis_dims = num_tensor_dims - num_explicit_dims

    # Replace ellipsis with appropriate number of slice(None)
    expanded_indices = indices[:ellipsis_pos] + (slice(None),) * num_ellipsis_dims + indices[ellipsis_pos + 1 :]

    return expanded_indices


class TensorMeta(FunsorMeta):
    def __call__(cls, data: torch.Tensor, indices: tuple[Funsor, ...] = ()) -> "Tensor":
        if len(indices) == 0:
            return data

        # Expand ellipsis in indices before processing
        indices = _expand_ellipsis(indices, data.shape)

        if all(isinstance(idx, Funsor) and not isinstance(idx, Tensor) for idx in indices):
            self = super().__call__(data, indices)
            assert isinstance(self, Tensor)
            return self

        # Separate funsor indices from non-funsor indices
        funsor_indices = []
        non_funsor_indices = []
        funsor_dims = []
        non_funsor_dims = []

        for i, idx in enumerate(indices):
            if isinstance(idx, Funsor) and not isinstance(idx, Tensor):
                funsor_indices.append(idx)
                funsor_dims.append(i)
            else:
                non_funsor_indices.append(idx)
                non_funsor_dims.append(i)

        remaining_dims = list(range(len(indices), data.ndim))
        full_permutation = funsor_dims + non_funsor_dims + remaining_dims

        permuted_data = data.permute(full_permutation)

        return Tensor(permuted_data, tuple(funsor_indices))[tuple(non_funsor_indices)]


class Tensor(Funsor, metaclass=TensorMeta):
    """
    Funsor representing a tensor.

    Example:

    >>> x = Tensor(torch.randn(5, 4, 3, 2), (Variable("i", int), Variable("j", int)))
    >>> x[2, 1]

    Args:
        data:
            The data of the tensor.
        indices:
            The indices of the tensor.
    """

    def __init__(self, data: torch.Tensor, indices: tuple[Funsor, ...] = ()) -> None:
        super().__init__("call_method", "__getitem__", (data, indices), {})
        self.data = data
        assert all(isinstance(idx, Funsor) for idx in indices)
        self.indices = indices

    @classmethod
    def make_hash_key(  # type: ignore[override]
        cls, data: torch.Tensor, indices: tuple[Funsor, ...] = ()
    ) -> tuple:
        return ("call_method", "__getitem__", (data, indices), (), None)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    # def __call__(self, *args, **kwargs):
    #     def materialize(x):
    #         if isinstance(x, Funsor) and not isinstance(x, (Variable, Tensor)):
    #             return self.materialize(x)
    #         return x

    #     args, kwargs = tree_map(materialize, (args, kwargs))

    #     return super().__call__(*args, **kwargs)

    def align(self, indices: tuple[Funsor, ...]) -> "Tensor":
        nodes = [index.node for index in indices]
        self_nodes = [index.node for index in self.indices]
        assert set(self_nodes) == set(nodes)

        # If indices are already in the same order, return self
        if self_nodes == nodes:
            return self

        # Create a permutation mapping from current indices to new indices
        permutation = []
        for node in nodes:
            old_pos = self_nodes.index(node)
            permutation.append(old_pos)

        permutation += list(range(len(permutation), self.data.ndim))

        # Permute the data dimensions according to the new indices order
        permuted_data = self.data.permute(permutation)

        return Tensor(permuted_data, indices)

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        # Delegate to superclass if any funsor arguments are not Tensor/Variable
        if tree_any(lambda x: isinstance(x, Funsor) and not isinstance(x, Tensor), [args, kwargs]):
            return super().__torch_function__(func, types, args, kwargs)

        return batched_call(func, *args, **kwargs)

    def __getattr__(self, k):
        if k == "_tensor_meta":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return TensorAttribute(self, k)

    def new_arange(self, name, *args, **kwargs):
        """
        Helper to create a named :func:`torch.arange` or :func:`np.arange` funsor.
        In some cases this can be replaced by a symbolic
        :class:`~funsor.terms.Slice` .

        :param str name: A variable name.
        :param int start:
        :param int stop:
        :param int step: Three args following :py:class:`slice` semantics.
        :param int dtype: An optional bounded integer type of this slice.
        :rtype: Tensor
        """
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        stop = min(dtype, max(start, stop))
        data = torch.arange(start, stop, step)
        indices = (Variable(name, int),)
        return Tensor(data, indices)

    # def materialize(self, x: Funsor) -> Funsor:
    #     """
    #     Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
    #     :class:`Tensor` by substituting :func:`arange` s into its free variables.

    #     :arg Funsor x: A funsor.
    #     :rtype: Funsor
    #     """
    #     assert isinstance(x, Funsor)
    #     if isinstance(x, (int, float, Tensor)):
    #         return x
    #     subs = {}
    #     for name, domain in x.inputs.items():
    #         if domain.dtype == torch.int64:
    #             subs[name] = self.new_arange(name, Variable(name, int).size)
    #     return x(**subs)

    def reduce(self, op, reduced_vars: frozenset["Variable"] | None = None):
        if reduced_vars is None:
            reduced_vars = self.indices
        reduced_nodes = [var.node for var in reduced_vars]
        reduced_dims = tuple(d for d, var in enumerate(self.indices) if var.node in reduced_nodes)
        new_indices = tuple(var for var in self.indices if var.node not in reduced_nodes)

        # Handle different PyTorch operations with different dim parameter requirements
        if op in [torch.prod]:
            # torch.prod expects individual dimensions, not a tuple
            if len(reduced_dims) == 0:
                data = self.data
            elif len(reduced_dims) == 1:
                data = op(self.data, dim=reduced_dims[0])
            else:
                # For multiple dimensions, reduce one by one
                data = self.data
                # Reduce in reverse order to maintain dimension indices
                for dim in sorted(reduced_dims, reverse=True):
                    data = op(data, dim=dim)
        else:
            data = op(self.data, dim=reduced_dims)

        return Tensor(data, new_indices)


class TensorAttribute(fx.Proxy):
    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    def __call__(self, *args, **kwargs):
        if tree_any(lambda x: isinstance(x, Funsor) and not isinstance(x, (Tensor, Variable)), [args, kwargs]):
            return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)
        else:
            return batched_call(getattr(torch.Tensor, self.attr), self.root, *args, **kwargs)


def batched_call(func, *args, **kwargs):
    # Collect unique variables from all tensor arguments, preserving order
    variables = []
    seen_nodes = set()

    for arg in tree_leaves(args):
        if isinstance(arg, Tensor):
            for var in arg.indices:
                if var.node not in seen_nodes:
                    seen_nodes.add(var.node)
                    variables.append(var)

    copy_indices = {}

    def get_in_dims(x, variable):
        if isinstance(x, Tensor):
            if x.node not in copy_indices:
                copy_indices[x.node] = tuple(x.indices)

            for i, idx in enumerate(copy_indices[x.node]):
                if idx.node == variable.node:
                    copy_indices[x.node] = tuple(copy_indices[x.node][:i] + copy_indices[x.node][i + 1 :])
                    return i
        return None

    # Handle scalar tensor arguments that cause issues with vmap
    unsqueeze_info = {}  # Maps tensor positions to whether they were unsqueezed

    def process_scalar_tensors(args):
        """Handle scalar tensor arguments for any operation that might cause vmap issues."""

        def process_arg(x):
            if isinstance(x, Tensor) and x.shape == ():
                # This is a scalar tensor that will cause issues with vmap
                # Unsqueeze it at the LAST dimension and track this change
                pos = id(x)
                unsqueeze_info[pos] = True
                unsqueezed_data = x.data.unsqueeze(-1)
                return Tensor(unsqueezed_data, x.indices)
            return x

        return tree_map(process_arg, args)

    # Process arguments to handle scalar tensors
    if func is operator.__getitem__ or func is torch.Tensor.__getitem__:
        processed_args = args[0], process_scalar_tensors(args[1])
    else:
        processed_args = args

    # Apply vmap for each variable dimension
    vectorized_func = func
    in_dims_list = []
    for variable in variables:
        # Map each argument to its dimension index for this variable
        in_dims = tree_map(
            lambda x: get_in_dims(x, variable),
            processed_args,
        )

        in_dims_list.append(in_dims)

    for in_dims in reversed(in_dims_list):
        vectorized_func = torch.vmap(vectorized_func, in_dims=in_dims)

    # Extract tensor data and apply the vectorized function
    data_args = tree_map(lambda x: x.data if isinstance(x, Tensor) else x, processed_args)
    result = vectorized_func(*data_args, **kwargs)

    # Squeeze back any dimensions we added for scalar tensors
    if unsqueeze_info:
        # If we unsqueezed any scalar tensors, we need to squeeze back
        # We unsqueezed at the last dimension, so squeeze the last dimension
        result = result.squeeze(len(variables))

    # Wrap result with variables in reverse order (outermost to innermost)
    return Tensor(result, tuple(variables))


for method in magic_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            if tree_any(lambda x: isinstance(x, Funsor) and not isinstance(x, Tensor), [args, kwargs]):
                tracer = args[0].tracer
                target = getattr(operator, method)
                return tracer.create_proxy("call_function", target, args, kwargs)
            else:
                return batched_call(getattr(operator, method), *args, **kwargs)

        impl.__name__ = method
        as_magic = f"__{method.strip('_')}__"
        setattr(Tensor, as_magic, impl)

    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f"__r{orig_method_name.strip('_')}__"

    def impl(self, rhs):
        if tree_any(lambda x: isinstance(x, Funsor) and not isinstance(x, Tensor), [self, rhs]):
            target = getattr(operator, orig_method_name)
            return self.tracer.create_proxy("call_function", target, (rhs, self), {})
        else:
            return batched_call(getattr(operator, orig_method_name), rhs, self)

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Tensor, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)


def align_tensor(new_indices: tuple[Tensor | torch.Tensor, ...], x: Tensor, expand: bool = False) -> torch.Tensor:
    r"""
    Permute and add dims to a tensor to match desired ``new_indices``.

    Args:
        new_indices: A target set of indices as tuple of Funsors.
        x: A :class:`Tensor`.
        expand: If False (default), set result size to 1 for any index
            of ``x`` not in ``new_indices``; if True expand to ``new_indices`` size.
    Returns:
        a :class:`torch.Tensor` that can be broadcast to other tensors with indices ``new_indices``.
    """
    if isinstance(x, torch.Tensor):
        return x

    data = x.data

    old_nodes = [idx.node for idx in x.indices]
    new_nodes = [idx.node for idx in new_indices]
    node_sizes = {node: data.shape[i] for i, node in enumerate(old_nodes)}

    # If indices are exactly the same (same nodes in same order), return the data as-is
    if old_nodes == new_nodes:
        return data

    # Permute squashed input dims.
    data = torch.permute(
        data,
        tuple(old_nodes.index(node) for node in new_nodes if node in old_nodes)
        + tuple(range(len(old_nodes), data.ndim)),
    )

    # Unsquash multivariate input dims by filling in ones.
    data = data.reshape(
        tuple(node_sizes[node] if node in old_nodes else 1 for node in new_nodes) + tuple(data.shape[len(old_nodes) :]),
    )

    return data


def align_tensors(*args):
    r"""
    Permute multiple tensors before applying a broadcasted op.
    """
    # Collect all unique indices from input tensors, preserving order
    indices_dict = {}

    for x in args:
        if isinstance(x, Tensor):
            indices_dict.update(dict.fromkeys(x.indices))

    indices = tuple(indices_dict)

    # Align all tensors to have the same indices
    tensors = [align_tensor(indices, x) for x in args]

    return indices, tensors
