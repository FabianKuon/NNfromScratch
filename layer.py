"""Layer implementation for neural net"""
from neuron import Neuron
from value_repr import Value


class Layer:
    """
    A layer implementation containing multiple neurons.
    """

    def __init__(self, dim_in: int, dim_out: int):
        self.neurons = [Neuron(dim_in) for _ in range(dim_out)]

    def __call__(self, features: list) -> list:
        outs = [n(features) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        """
        Extract all parameters associated with the specific layer.
        """
        return [parameter for neuron in self.neurons for parameter in neuron.parameters]
