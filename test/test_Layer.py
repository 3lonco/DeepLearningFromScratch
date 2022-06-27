# Add path for CI/CD tool
from locale import normalize
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../script/")
)
import numpy as np
import pytest
import some_Layer
import load_mnist


def test_sigmoidLayer():
    sigmoid_layer = some_Layer.Sigmoid()
    # create the values only with nagative values
    x = np.array([-100.0, 100, -1.5])
    y = sigmoid_layer.forward(x)
    assert y.all() >= 0

def test_ReLULayer():
    ReLu_layer=some_Layer.Relu()
    # create the values only with nagative values
    x = np.array([-100.0, 100, -1.5])
    y = ReLu_layer.forward(x)
    assert y.all() >= 0


def test_MulLayer_1():
    apple = 100
    apple_num = 2
    tax = 1.1
    # layer
    mul_apple_Layer = some_Layer.MulLayer()
    mul_tax_Layer = some_Layer.MulLayer()

    # forward
    apple_price = mul_apple_Layer.forward(apple, apple_num)  # 200
    price = mul_tax_Layer.forward(apple_price, tax)  # 220

    # backward
    dprice = 1
    dapple_price, dtax = mul_tax_Layer.backward(dprice)
    dapple, dapple_num = mul_apple_Layer.backward(dapple_price)

    solve = np.array([apple_price, price, dtax, dapple, dapple_num])
    ans = np.array([200, 220, 200, 2.2, 110])
    np.testing.assert_almost_equal(solve, ans)


def test_MulLayer_2():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # layer
    mul_apple_Layer = some_Layer.MulLayer()
    mul_orange_Layer = some_Layer.MulLayer()
    add_apple_orange_Layer = some_Layer.AddLayer()
    mul_tax_Layer = some_Layer.MulLayer()

    # forward
    apple_price = mul_apple_Layer.forward(apple, apple_num)  # 200
    orange_price = mul_orange_Layer.forward(orange, orange_num)  # 450
    all_price = add_apple_orange_Layer.forward(apple_price, orange_price)  # 650
    price = mul_tax_Layer.forward(all_price, tax)  # 715

    # backward
    dprice = 1
    dall_price, dtax = mul_tax_Layer.backward(dprice)  # 1.1 650
    dapple_price, dorange_price = add_apple_orange_Layer.backward(
        dall_price
    )  # 1.1 #1.1
    dapple, dapple_num = mul_apple_Layer.backward(dapple_price)  # 2.2,110
    dorange, dorange_num = mul_orange_Layer.backward(dorange_price)  # 3.3,165

    sol = np.array(
        [
            apple_price,
            orange_price,
            all_price,
            price,
            dall_price,
            dtax,
            dapple_price,
            dorange_price,
            dapple,
            dapple_num,
            dorange,
            dorange_num,
        ]
    )
    ans = np.array([200, 450, 650, 715, 1.1, 650, 1.1, 1.1, 2.2, 110, 3.3, 165])
    np.testing.assert_almost_equal(sol, ans)


def test_minibatch_gradient_decent():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(
        normalize=True, one_hot_label=True
    )
    train_loss_list = []

    # Hyper parameter
    iters_num = 2
    train_size = x_train.shape[0]
    batch_size = 2
    learning_rate = 0.1
    network = some_Layer.TwoLayerNet(input_size=784, hidden_size=3, output_size=10)
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
    net = some_Layer.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)
    x = np.random.rand(100, 784)  # ダミーの入力データ
    print("x,shape", x.shape)
    y = net.predict(x)
    print("predict(x).shape:", y.shape)
    t = np.random.rand(100, 10)
