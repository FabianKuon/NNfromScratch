"""Training of a mlp with a simple data example"""
import matplotlib.pyplot as plt
from Model.multiLayerPerceptron import MLP


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
    # n() calls the implemented __call__ method of the mlp, since n is a mlp instance
    loss_overall = []
    step = []
    for k in range(20):
        # forward pass
        ypred = [model(x) for x in features]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(labels, ypred))

        # backward pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in model.parameters():
            p.data += -0.1 * p.grad

        loss_overall.append(loss.data)
        step.append(k)

    plt.title('Loss function')
    plt.plot(loss_overall)
    plt.show()
