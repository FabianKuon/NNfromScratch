"""Multi-Layer-Perceptron implementation"""
from Model.layer import Layer


class MLP:
    """
    Multi-Layer-Perceptron
    """

    def __init__(self, dim_in: int, dim_outs: list[int]):
        """
        param: dim_in   dimension of input features
        param: dim_outs list of neurons at each layer
        """
        size = [dim_in] + dim_outs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(dim_outs))]

    def __call__(self, features: list) -> list:
        for layer in self.layers:
            activation = layer(features)
        return activation

    def parameters(self):
        """
        List of all parameters belonging to the MLP.
        """
        return [parameter for layer in self.layers for parameter in layer.parameters]
