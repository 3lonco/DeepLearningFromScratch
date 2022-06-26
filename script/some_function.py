import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # For Overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def step_function(x):
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def identity_function(x):
    if type(x) == str:
        raise TypeError(x, "This is a string!")
    return x


def cross_entropy_error(y, t):
    # Where, we plus delta because np.log(0) is negative infinity that stop a model from proceeding this process.
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
