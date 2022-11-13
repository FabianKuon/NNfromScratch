"""Layer implementation for neural net"""
from Model.neuron import Neuron
from Model.value_repr import Value


class Layer:
    """
    A layer implementation containing multiple neurons.
    """

    def __init__(self, dim_in: int, dim_out: int, activation_func: str):
        self.neurons = [Neuron(dim_in, activation_func)
                        for _ in range(dim_out)]

    def __call__(self, features: list[Value]) -> list:
        outs = [n(features) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        """
        Extract all parameters associated with the specific layer.
        """
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]
