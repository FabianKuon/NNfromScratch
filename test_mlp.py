"""Training of a mlp with a simple data example"""
from multiLayerPerceptron import MLP


if __name__ == '__main__':
    # a multi-layer-perceptron with input dimension 3,
    # first and second layer dimension 4 and outpit dimension 1
    n = MLP(3, [4, 4, 1])
    features = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    labels = [1.0, -1.0, -1.0, 1.0]
    # n() calls the implemented __call__ method of the mlp, since n is a mlp instance
    predictions = [n(x) for x in features]
    loss = sum((yout - ygt)**2 for yout, ygt in zip(predictions, labels))

    loss.backward()
    print(n.layers[0].neurons[0].weights[0].grad)
