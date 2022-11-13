"""Training of a mlp with a simple data example"""
from Model.multi_layer_perceptron import MLP


if __name__ == '__main__':
    # a multi-layer-perceptron with input dimension 3,
    # first and second layer dimension 4 and outpit dimension 1
    model = MLP(3, [4, 4, 1])
    features = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    labels = [1.0, -1.0, -1.0, 1.0]

    model.train(features, labels, 1000, True)
    model.visualize(labels)
