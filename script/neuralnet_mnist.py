import pickle
import numpy as np
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../script/")
)
import load_mnist
import some_function


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(
        flatten=True, normalize=False, one_hot_label=False
    )
    return x_test, t_test


def init_network():
    path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../script/")
    pkl = path + "/sample_weight.pkl"
    with open(pkl, "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = some_function.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = some_function.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = some_function.softmax(a3)
    return y
