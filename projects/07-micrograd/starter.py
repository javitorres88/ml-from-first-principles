"""Starter: a mini reverse-mode autodiff engine."""
import numpy as np


class Tensor:
    """A value plus the information needed to backpropagate through it."""

    def __init__(self, data, parents=(), backward_fn=None, requires_grad=False):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def backward(self):
        """Topological sort, then a reverse sweep with gradient accumulation."""
        raise NotImplementedError


class SGD:
    def __init__(self, params, lr=0.01):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


if __name__ == "__main__":
    print("target: match PyTorch gradients to 1e-6 on random inputs")
