import numpy as np
from script.some_function import cross_entropy_error, softmax


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # Initialize with Gaussian Distribution

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
