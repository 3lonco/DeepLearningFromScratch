import sys, os

sys.path.append(os.pardir)
import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(
    flatten=True, normalize=False
)
