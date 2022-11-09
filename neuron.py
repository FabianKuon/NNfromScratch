"""Class representation for neurons"""
from numpy import random
from value_repr import Value


class Neuron:
    """
    The Neuron class used for neural nets
    """

    def __init__(self, dimension: int) -> None:
        self.weights = [Value(random.uniform(-1, 1))
                        for _ in range(dimension)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, data: list) -> list:
        act = sum((wi*xi for wi, xi in zip(self.weights, data)), self.bias)
        out = act.tanh()
        return out
