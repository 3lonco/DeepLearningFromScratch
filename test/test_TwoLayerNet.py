# Add path for CI/CD tool
from locale import normalize
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../script/")
)
import numpy as np
import pytest
import TwoLayerNet
import load_mnist


def test_minibatch_gradient_decent():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(
        normalize=True, one_hot_label=True
    )
    train_loss_list = []

    # Hyper parameter
    iters_num = 2
    train_size = x_train.shape[0]
    batch_size = 10
    learning_rate = 0.1
    network = TwoLayerNet.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        # Get mini batch
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # Calculate gradient
        grad = network.numerical_gradient(x_batch, t_batch)
        # grad = network.gradient(x_batch,t_batch)# Faster version.Implement later

        # Renew parameters
        for key in ("W1", "b1", "W2", "b2"):
          network.params[key] -= learning_rate * grad[key]

        # Record the learning process
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)


def test_TwoLayerNet():
    net = TwoLayerNet.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)
    x = np.random.rand(100, 784)  # ダミーの入力データ
    print("x,shape", x.shape)
    y = net.predict(x)
    print("predict(x).shape:", y.shape)
    t = np.random.rand(100, 10)
