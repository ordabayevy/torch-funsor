# torch-funsor

A PyTorch-native reimplementation of the [Funsor](https://arxiv.org/abs/1910.10775) library for tensor-like operations on functions and distributions.

## Overview

**torch-funsor** is a reimplementation of the original [Funsor library](https://github.com/pyro-ppl/funsor/tree/master) with native PyTorch support. This allows PyTorch tensors and Funsor objects to be directly mixed and used together seamlessly.

## Key Features

- **Native PyTorch Integration**: Direct interoperability between PyTorch tensors and Funsor objects
- **torch.fx Backend**: Leverages PyTorch's functional transformation framework for efficient computation
- **Tensor-like Operations**: Supports function and distribution manipulations with familiar tensor semantics

## About Funsor

Funsor is a tensor-like library designed for functions and distributions, enabling sophisticated probabilistic programming and symbolic computation. For more details, see the [original paper](https://arxiv.org/abs/1910.10775).