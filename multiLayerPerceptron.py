"""Multi-Layer-Perceptron implementation"""
from layer import Layer


class MLP:
    """
    Multi-Layer-Perceptron
    """

    def __init__(self, dim_in: int, dim_outs: list[int]) -> None:
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
