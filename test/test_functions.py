# Add path for CI/CD tool
import sys
import os
from matplotlib.pyplot import get
from script.neuralnet_mnist import get_data, init_network, predict

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../script/")
)
import some_function
import numpy as np
import pytest
import load_mnist

def test_cross_entropy_error_1():
    # normal test
    # define 2 as correct
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    ans = some_function.cross_entropy_error(np.array(y), np.array(t))
    #print(ans)
    #np.testing.assert_array_almost_equal(ans, 0.51082545, decimal=1)


def test_mean_squared_error():
    # define 2 as correct
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    ans = some_function.mean_squared_error(np.array(y), np.array(t))
    np.testing.assert_array_almost_equal(ans, 0.097500000000, decimal=3)


def test_neural_net():
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


def test_neural_net_test():
    x, t = get_data()
    batch_size = 100  # the number of batch
    network = init_network()
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i : i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i : i + batch_size])


def test_step_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = some_function.step_function(x)
    # All figures are expected to be larger than equal zero
    assert y.all() >= 0


def test_step_function_type_error():
    # Raise TypeError when you give str into step_function
    with pytest.raises(TypeError) as e:
        some_function.step_function("a")


def test_step_function_error():
    a = some_function.step_function(1)
    # 1を入れたら1が返ってくる
    assert a == 1


def test_sigmoid_1():
    # create the values only with nagative values
    x = np.array([-100.0, 100, -1.5])
    y = some_function.sigmoid(x)
    assert y.all() >= 0


def test_sigmoid_assertion_error():
    x = np.array([-1.0, 1.0, 2.0])
    y = some_function.sigmoid(x)
    # Create the theorotical vaulues
    ans = np.array([0.26894142, 0.73105858, 0.88079707])
    np.testing.assert_array_almost_equal(y, ans, decimal=3)


def test_sigmoid_assertion_error():
    x = np.array([-1.0, 1.0, 2.0])
    y = some_function.sigmoid(x)
    ans = np.array([3.26894142, 0.73105858, 0.88079707])
    # Raise AssetionError by putting wrong numbers
    with pytest.raises(AssertionError) as e:
        np.testing.assert_array_almost_equal(y, ans, decimal=1)


def test_sigmoid_type_error():
    with pytest.raises(TypeError) as e:
        some_function.sigmoid("a")


def test_relu_negative():
    x = np.array([-100, 0, 1.0])
    y = some_function.relu(x)
    assert y.all() <= 0


def test_relu_positive():
    x = np.array([0.1, 100, 1.0])
    y = some_function.relu(x)
    assert y.all() > 0


def test_identity_function():
    x = np.array([1, 2, 0.1])
    y = some_function.identity_function(x)
    assert y.all() == x.all()


def test_identity_function():
    with pytest.raises(TypeError) as e:
        some_function.identity_function("a")


def test_softmax():
    a = np.array([0.3, 2.9, 4.0])
    y = some_function.softmax(a)
    # Check the output of softmax is the same as the following numbers
    ans = np.array([0.01821127, 0.24519181, 0.73659691])
    assert y.all() == ans.all()


def test_softmax_sum():
    arr = np.random.rand(1000)
    sum = np.sum(some_function.softmax(arr))
    # Check if the sum of an randam array is equal to 1.
    assert pytest.approx(sum, 0.00001) == 1.0


def test_load():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(
        flatten=True, normalize=False
    )
    assert t_train.shape[0] == 60000
