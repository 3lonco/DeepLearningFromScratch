import numpy as np
import pytest


def test_dot():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.dot(A, B)
    Ans = np.array([[19, 22], [43, 50]])
    assert Ans.all() == C.all()
