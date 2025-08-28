# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from funsor import ShapedTensor, Tensor, Variable
from funsor.tensor import align_tensors
from funsor.utilities.testing import (
    OPERATOR_BINARY_OPS,
    OPERATOR_BOOLEAN_OPS,
    TORCH_BINARY_OPS,
    TORCH_BOOLEAN_OPS,
    assert_equiv,
    check_funsor,
    eval_shape,
)


def test_cons_hash():
    i = Variable("i", int)
    x = torch.randn((3, 3))
    assert Tensor(x, (i,)) is Tensor(x, (i,))
    assert Tensor(x, (i,)) is not Tensor(x.clone(), (i,))


def test_indexing():
    data = torch.randn((4, 5))
    i = Variable("i", int)
    j = Variable("j", int)
    indices = (i, j)
    inputs = {"i": int, "j": int}

    x = Tensor(data, indices)
    check_funsor(x, inputs, torch.float32, data)

    assert x() is x
    check_funsor(x(1), {"j": int}, torch.float32, data[1])
    check_funsor(x(1, 2), {}, torch.float32, data[1, 2])
    check_funsor(x(1, j=2), {}, torch.float32, data[1, 2])
    check_funsor(x(i=1), {"j": int}, torch.float32, data[1])
    check_funsor(x(i=1, j=2), {}, torch.float32, data[1, 2])
    check_funsor(x(j=2), {"i": int}, torch.float32, data[:, 2])


def test_advanced_indexing_shape():
    I, J = 4, 4
    x = Tensor(torch.randn((I, J)), (Variable("i", int), Variable("j", int)))
    m = Tensor(torch.tensor([2, 3]), (Variable("m", int),))
    n = Tensor(torch.tensor([0, 1, 1]), (Variable("n", int),))
    assert x.data.shape == (I, J)

    check_funsor(x(i=m), {"j": int, "m": int}, torch.float32)
    check_funsor(x(i=m, j=n), {"m": int, "n": int}, torch.float32)
    check_funsor(x(i=n), {"j": int, "n": int}, torch.float32)
    check_funsor(x(j=m), {"i": int, "m": int}, torch.float32)
    check_funsor(x(j=m, i=n), {"m": int, "n": int}, torch.float32)
    check_funsor(x(j=n), {"i": int, "n": int}, torch.float32)
    check_funsor(x(m), {"j": int, "m": int}, torch.float32)
    check_funsor(x(m, j=n), {"m": int, "n": int}, torch.float32)
    check_funsor(x(m, n), {"m": int, "n": int}, torch.float32)
    check_funsor(x(n), {"j": int, "n": int}, torch.float32)
    check_funsor(x(n, m), {"m": int, "n": int}, torch.float32)


@pytest.mark.parametrize("output_shape", [(), (7,), (3, 2)])
def test_advanced_indexing_tensor(output_shape):
    #      u   v
    #     / \ / \
    #    i   j   k
    #     \  |  /
    #      \ | /
    #        x
    x = Tensor(torch.randn((2, 3, 4) + output_shape), (Variable("i", int), Variable("j", int), Variable("k", int)))
    i = Tensor(torch.randint(0, 2, (5,)), (Variable("u", int),))
    j = Tensor(torch.randint(0, 3, (6, 5)), (Variable("v", int), Variable("u", int)))
    k = Tensor(torch.randint(0, 4, (6,)), (Variable("v", int),))

    expected_data = torch.empty((5, 6) + output_shape)
    for u in range(5):
        for v in range(6):
            expected_data[u, v] = x.data[i.data[u], j.data[v, u], k.data[v]]
    expected = Tensor(expected_data, (Variable("u", int), Variable("v", int)))

    assert_equiv(expected, x(i, j, k))
    assert_equiv(expected, x(i=i, j=j, k=k))

    assert_equiv(expected, x(i=i, j=j)(k=k))
    assert_equiv(expected, x(j=j, k=k)(i=i))
    assert_equiv(expected, x(k=k, i=i)(j=j))

    assert_equiv(expected, x(i=i)(j=j, k=k))
    assert_equiv(expected, x(j=j)(k=k, i=i))
    assert_equiv(expected, x(k=k)(i=i, j=j))

    assert_equiv(expected, x(i=i)(j=j)(k=k))
    assert_equiv(expected, x(i=i)(k=k)(j=j))
    assert_equiv(expected, x(j=j)(i=i)(k=k))
    assert_equiv(expected, x(j=j)(k=k)(i=i))
    assert_equiv(expected, x(k=k)(i=i)(j=j))
    assert_equiv(expected, x(k=k)(j=j)(i=i))


@pytest.mark.skip(reason="skipping advanced indexing lazy test")
@pytest.mark.parametrize("output_shape", [(), (7,), (3, 2)])
def test_advanced_indexing_lazy(output_shape):
    x = Tensor(torch.randn((2, 3, 4) + output_shape), (Variable("i", int), Variable("j", int), Variable("k", int)))
    u = Variable("u", int)
    v = Variable("v", int)
    i = 1 - u
    j = 2 - v
    k = u + v

    expected_data = torch.empty((2, 3) + output_shape)
    i_data = x.materialize(i).data
    j_data = x.materialize(j).data
    k_data = x.materialize(k).data
    for u in range(2):
        for v in range(3):
            expected_data[u, v] = x.data[i_data[u], j_data[v], k_data[u, v]]
    expected = Tensor(expected_data, (Variable("u", int), Variable("v", int)))

    assert_equiv(expected, x(i, j, k))
    assert_equiv(expected, x(i=i, j=j, k=k))

    assert_equiv(expected, x(i=i, j=j)(k=k))
    assert_equiv(expected, x(j=j, k=k)(i=i))
    assert_equiv(expected, x(k=k, i=i)(j=j))

    assert_equiv(expected, x(i=i)(j=j, k=k))
    assert_equiv(expected, x(j=j)(k=k, i=i))
    assert_equiv(expected, x(k=k)(i=i, j=j))

    assert_equiv(expected, x(i=i)(j=j)(k=k))
    assert_equiv(expected, x(i=i)(k=k)(j=j))
    assert_equiv(expected, x(j=j)(i=i)(k=k))
    assert_equiv(expected, x(j=j)(k=k)(i=i))
    assert_equiv(expected, x(k=k)(i=i)(j=j))
    assert_equiv(expected, x(k=k)(j=j)(i=i))


@pytest.mark.parametrize(
    "op",
    [
        "abs",
        "ceil",
        "floor",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "acos",
        "cos",
        "sigmoid",
        "tanh",
        "sin",
        "atanh",
        "asin",
    ],
)
@pytest.mark.parametrize("dims", [("a",), ("a", "b")])
@pytest.mark.parametrize("kind", ["function", "method"])
def test_torch_unary(op, dims, kind):
    sizes = {"a": 3, "b": 4}
    shape = tuple(sizes[d] for d in dims)
    inputs = {d: ShapedTensor[int, ()] for d in dims}
    indices = tuple(Variable(d, int) for d in dims)
    data = torch.rand(shape) + 0.5

    expected_data = getattr(torch, op)(data)

    x = Tensor(data, indices)
    if kind == "function":
        actual = getattr(torch, op)(x)
    else:
        actual = getattr(x, op)()
    check_funsor(actual, inputs, torch.float32, expected_data)


@pytest.mark.parametrize("dims2", [(), ("a",), ("b", "a"), ("b", "c", "a")])
@pytest.mark.parametrize("dims1", [(), ("a",), ("a", "b"), ("b", "a", "c")])
@pytest.mark.parametrize("binary_op", OPERATOR_BINARY_OPS + OPERATOR_BOOLEAN_OPS + TORCH_BINARY_OPS + TORCH_BOOLEAN_OPS)
def test_binary_funsor_funsor(binary_op, dims1, dims2):
    sizes = {"a": 3, "b": 4, "c": 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)

    # Generate test data
    data1 = torch.rand(shape1)
    data2 = torch.rand(shape2)
    if binary_op in OPERATOR_BOOLEAN_OPS + TORCH_BOOLEAN_OPS:
        data1 = (data1 > 0.5).to(torch.bool)
        data2 = (data2 > 0.5).to(torch.bool)

    # Create Tensor objects
    indices1 = tuple(Variable(d, int) for d in dims1)
    indices2 = tuple(Variable(d, int) for d in dims2)
    x1 = Tensor(data1, indices1)
    x2 = Tensor(data2, indices2)

    # Expected output
    aligned_indices, aligned_tensors = align_tensors(x1, x2)
    expected_inputs = {d.name: int for d in aligned_indices}
    expected_data = binary_op(*aligned_tensors)
    expected_dtype = expected_data.dtype

    actual = binary_op(x1, x2)
    check_funsor(actual, expected_inputs, expected_dtype, expected_data)


@pytest.mark.parametrize("output_shape2", [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize("output_shape1", [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize("inputs2", [(), ("a",), ("b", "a"), ("b", "c", "a")], ids=str)
@pytest.mark.parametrize("inputs1", [(), ("a",), ("a", "b"), ("b", "a", "c")], ids=str)
def test_binary_broadcast(inputs1, inputs2, output_shape1, output_shape2):
    sizes = {"a": 4, "b": 5, "c": 6}

    # Create variables and tensor shapes for inputs1
    indices1 = tuple(Variable(k, int) for k in inputs1)
    shape1 = tuple(sizes[k] for k in inputs1) + output_shape1
    data1 = torch.randn(shape1)
    x1 = Tensor(data1, indices1)

    # Create variables and tensor shapes for inputs2 (fix bug: was using inputs1)
    indices2 = tuple(Variable(k, int) for k in inputs2)
    shape2 = tuple(sizes[k] for k in inputs2) + output_shape2
    data2 = torch.randn(shape2)
    x2 = Tensor(data2, indices2)

    # Test broadcasting addition
    actual = x1 + x2

    # Check that the result has expected output shape type
    expected_output = eval_shape(torch.add, x1, x2)
    assert actual.dtype == expected_output.dtype
    assert actual.shape == expected_output.shape

    # Test evaluation with specific values
    block = {"a": 1, "b": 2, "c": 3}

    if isinstance(actual, Tensor):
        actual_block = actual(**{k: v for k, v in block.items() if k in actual.inputs})
    else:
        actual_block = actual

    if isinstance(x1, Tensor):
        x1_block = x1(**{k: v for k, v in block.items() if k in x1.inputs})
    else:
        x1_block = x1

    if isinstance(x2, Tensor):
        x2_block = x2(**{k: v for k, v in block.items() if k in x2.inputs})
    else:
        x2_block = x2

    # Verify the computation is correct by comparing the tensor data
    expected_data = x1_block.data + x2_block.data
    assert torch.allclose(actual_block.data, expected_data, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("output_shape2", [(2,), (2, 5), (4, 2, 5)], ids=str)
@pytest.mark.parametrize("output_shape1", [(2,), (3, 2), (4, 3, 2)], ids=str)
@pytest.mark.parametrize("inputs2", [(), ("a",), ("b", "a"), ("b", "c", "a")], ids=str)
@pytest.mark.parametrize("inputs1", [(), ("a",), ("a", "b"), ("b", "a", "c")], ids=str)
def test_matmul(inputs1, inputs2, output_shape1, output_shape2):
    sizes = {"a": 6, "b": 7, "c": 8}

    # Create variables and tensor shapes for inputs1
    indices1 = tuple(Variable(k, int) for k in inputs1)
    shape1 = tuple(sizes[k] for k in inputs1) + output_shape1
    data1 = torch.randn(shape1)
    x1 = Tensor(data1, indices1)

    # Create variables and tensor shapes for inputs2 (fix bug: was using inputs1)
    indices2 = tuple(Variable(k, int) for k in inputs2)
    shape2 = tuple(sizes[k] for k in inputs2) + output_shape2
    data2 = torch.randn(shape2)
    x2 = Tensor(data2, indices2)

    actual = x1 @ x2
    expected_output = eval_shape(torch.matmul, x1, x2)
    assert actual.dtype == expected_output.dtype
    assert actual.shape == expected_output.shape

    block = {"a": 1, "b": 2, "c": 3}
    if isinstance(actual, Tensor):
        actual_block = actual(**{k: v for k, v in block.items() if k in actual.inputs})
    else:
        actual_block = actual
    if isinstance(x1, Tensor):
        x1_block = x1(**{k: v for k, v in block.items() if k in x1.inputs})
    else:
        x1_block = x1
    if isinstance(x2, Tensor):
        x2_block = x2(**{k: v for k, v in block.items() if k in x2.inputs})
    else:
        x2_block = x2
    expected_block = x1_block @ x2_block
    assert torch.allclose(actual_block.data, expected_block.data, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("scalar", [0.5, -0.5])
@pytest.mark.parametrize("dims", [(), ("a",), ("a", "b"), ("b", "a", "c")])
@pytest.mark.parametrize("binary_op", OPERATOR_BINARY_OPS + OPERATOR_BOOLEAN_OPS + TORCH_BINARY_OPS + TORCH_BOOLEAN_OPS)
def test_binary_funsor_scalar(binary_op, dims, scalar):
    sizes = {"a": 3, "b": 4, "c": 5}
    shape = tuple(sizes[d] for d in dims)
    indices = tuple(Variable(d, int) for d in dims)
    inputs = {d: int for d in dims}
    data1 = torch.rand(shape)
    if binary_op in OPERATOR_BOOLEAN_OPS + TORCH_BOOLEAN_OPS:
        data1 = (data1 > 0.5).to(torch.bool)
        scalar = scalar > 0.0
    if binary_op in TORCH_BINARY_OPS + TORCH_BOOLEAN_OPS:
        scalar = torch.tensor(scalar)

    expected_data = binary_op(data1, scalar)
    expected_dtype = expected_data.dtype

    x1 = Tensor(data1, indices)
    actual = binary_op(x1, scalar)
    check_funsor(actual, inputs, expected_dtype, expected_data)


@pytest.mark.parametrize("scalar", [0.5, -0.5])
@pytest.mark.parametrize("dims", [(), ("a",), ("a", "b"), ("b", "a", "c")])
@pytest.mark.parametrize("binary_op", OPERATOR_BINARY_OPS + OPERATOR_BOOLEAN_OPS + TORCH_BINARY_OPS + TORCH_BOOLEAN_OPS)
def test_binary_scalar_funsor(binary_op, dims, scalar):
    sizes = {"a": 3, "b": 4, "c": 5}
    shape = tuple(sizes[d] for d in dims)
    indices = tuple(Variable(d, int) for d in dims)
    inputs = {d: int for d in dims}
    data1 = torch.rand(shape)
    if binary_op in OPERATOR_BOOLEAN_OPS + TORCH_BOOLEAN_OPS:
        data1 = (data1 > 0.5).to(torch.bool)
        scalar = scalar > 0.0
    if binary_op in TORCH_BINARY_OPS + TORCH_BOOLEAN_OPS:
        scalar = torch.tensor(scalar)

    expected_data = binary_op(scalar, data1)
    expected_dtype = expected_data.dtype

    x1 = Tensor(data1, indices)
    actual = binary_op(scalar, x1)
    check_funsor(actual, inputs, expected_dtype, expected_data)


@pytest.mark.parametrize("batch_shape", [(), (5,), (4, 3)])
@pytest.mark.parametrize(
    "old_shape,new_shape",
    [
        ((), ()),
        ((), (1,)),
        ((2,), (2, 1)),
        ((2,), (1, 2)),
        ((6,), (2, 3)),
        ((6,), (2, 1, 3)),
        ((2, 3, 2), (3, 2, 2)),
        ((2, 3, 2), (2, 2, 3)),
    ],
)
def test_reshape(batch_shape, old_shape, new_shape):
    indices = tuple(Variable(d, int) for d, _ in zip("abc", batch_shape))
    old = Tensor(torch.randn(batch_shape + old_shape), indices)
    assert_equiv(old.reshape(old.shape), old)

    new = old.reshape(new_shape)
    new_inputs = old.inputs if isinstance(old, Tensor) else {}
    new_output = ShapedTensor[old.dtype, new_shape]
    check_funsor(new, new_inputs, new_output)

    old2 = new.reshape(old_shape)
    assert_equiv(old2, old)


def test_getitem_number_1_inputs():
    data = torch.randn((3, 5, 4, 3, 2))
    indices = (Variable("i", int),)
    x = Tensor(data, indices)
    assert_equiv(x[2], Tensor(data[:, 2], indices))
    assert_equiv(x[:, 1], Tensor(data[:, :, 1], indices))
    assert_equiv(x[2, 1], Tensor(data[:, 2, 1], indices))
    assert_equiv(x[2, :, 1], Tensor(data[:, 2, :, 1], indices))
    assert_equiv(x[3, ...], Tensor(data[:, 3, ...], indices))
    assert_equiv(x[3, 2, ...], Tensor(data[:, 3, 2, ...], indices))
    assert_equiv(x[..., 1], Tensor(data[..., 1], indices))
    assert_equiv(x[..., 2, 1], Tensor(data[..., 2, 1], indices))
    assert_equiv(x[3, ..., 1], Tensor(data[:, 3, ..., 1], indices))


def test_getitem_number_2_inputs():
    data = torch.randn((3, 4, 5, 4, 3, 2))
    indices = (Variable("i", int), Variable("j", int))
    x = Tensor(data, indices)
    assert_equiv(x[2], Tensor(data[:, :, 2], indices))
    assert_equiv(x[:, 1], Tensor(data[:, :, :, 1], indices))
    assert_equiv(x[2, 1], Tensor(data[:, :, 2, 1], indices))
    assert_equiv(x[2, :, 1], Tensor(data[:, :, 2, :, 1], indices))
    assert_equiv(x[3, ...], Tensor(data[:, :, 3, ...], indices))
    assert_equiv(x[3, 2, ...], Tensor(data[:, :, 3, 2, ...], indices))
    assert_equiv(x[..., 1], Tensor(data[..., 1], indices))
    assert_equiv(x[..., 2, 1], Tensor(data[..., 2, 1], indices))
    assert_equiv(x[3, ..., 1], Tensor(data[:, :, 3, ..., 1], indices))


def test_getitem_variable():
    data = torch.randn((5, 4, 3, 2))
    i = Variable("i", int)
    j = Variable("j", int)
    k = Variable("k", int)
    l = Variable("l", int)
    assert data[i] is Tensor(data, (i,))
    assert data[i, j] is Tensor(data, (i, j))
    assert data[i, j, k] is Tensor(data, (i, j, k))
    assert data[i, j, k, l] is Tensor(data, (i, j, k, l))
    assert_equiv(data[:, j], Tensor(data, (slice(None), j)))
    assert_equiv(data[i, :, k], Tensor(data, (i, slice(None), k)))
    assert_equiv(data[..., j, k, l], Tensor(data, (Ellipsis, j, k, l)))
    assert_equiv(data[..., k, l], Tensor(data, (Ellipsis, k, l)))
    assert_equiv(data[..., l], Tensor(data, (Ellipsis, l)))


def test_getitem_tensor():
    x = torch.randn((5, 4, 3, 2))
    i = Variable("i", int)
    i_size = 5
    j = Variable("j", int)
    j_size = 4
    k = Variable("k", int)
    k_size = 3
    m = Variable("m", int)

    y = torch.randint(0, 5, ())
    assert_equiv(x[i](i=y), x[y])

    y = torch.randint(0, 4, ())
    assert_equiv(x[:, j](j=y), x[:, y])

    y = torch.randint(0, 3, ())
    assert_equiv(x[:, :, k](k=y), x[:, :, y])

    y = torch.randint(0, 2, ())
    assert_equiv(x[:, :, :, m](m=y), x[:, :, :, y])

    y = Tensor(torch.randint(0, j_size, (i_size,)), (i,))
    assert_equiv(x[i, j](j=y), x[i, y])

    y = Tensor(torch.randint(0, k_size, (i_size, j_size)), (i, j))
    assert_equiv(x[i, j, k](k=y), x[i, j, y])


REDUCE_OPS = [
    torch.sum,
    torch.prod,
    torch.all,
    torch.any,
    torch.logsumexp,
    torch.amin,
    torch.amax,
]


@pytest.mark.parametrize("dims", [("a",), ("a", "b"), ("b", "a", "c")])
@pytest.mark.parametrize("op", REDUCE_OPS, ids=str)
def test_reduce_all(dims, op):
    sizes = {"a": 3, "b": 4, "c": 5}
    shape = tuple(sizes[d] for d in dims)
    indices = tuple(Variable(d, int) for d in dims)
    data = torch.rand(shape)
    if op in [torch.all, torch.any]:
        data = data > 0.5

    # Compute expected result based on operation requirements
    if op in [torch.logsumexp]:
        # These operations require explicit dimensions
        all_dims = tuple(range(len(shape)))
        expected_data = op(data, dim=all_dims)
    else:
        expected_data = op(data)

    x = Tensor(data, indices)
    actual = x.reduce(op)
    check_funsor(actual, {}, expected_data.dtype, expected_data)
