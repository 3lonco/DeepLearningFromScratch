import numpy as np


def gradient_descent(f, init_x, lr=00.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # Generate the same array as x
    for idx in range(x.size):
        tmp_val = x[idx]

        # Calculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # Caculate f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


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
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
