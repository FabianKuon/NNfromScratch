"""Class representation for neurons"""
from numpy import random
from Model.value_repr import Value
from Model.activation_functions_enum import ValidActivationFunctions as vaf


class Neuron:
    """
    The Neuron class used for neural nets
    """

    def __init__(self, dimension: int, activation_func: vaf) -> None:
        self.weights = [Value(random.uniform(-1, 1))
                        for _ in range(dimension)]
        self.baises = Value(random.uniform(-1, 1))
        self.activation_func = activation_func

    def __call__(self, data: list[float]) -> list:
        if isinstance(data, float):
            act = (self.weights[0] * data).tanh()
        else:
            act = sum((wi*xi for wi, xi in zip(self.weights, data)), self.baises)
        match self.activation_func:
            case vaf.TANH:
                out = act.tanh()
            case vaf.RELU:
                out = act.relu()
            case _:
                out = act.relu()
        return out

    def parameters(self) -> list[Value]:
        """
        Returns a list of all parameters on the specific neuron
        """
        return self.weights + [self.baises]
