
import numpy as np
from SparseGroupLasso.linalg import dot
from numpy.testing import assert_almost_equal


def test_dot():
    X = np.random.randn(10, 8)
    beta = np.random.randn(8)
    dot1 = dot(X, beta)
    dot2 = np.dot(X, beta)
    assert_almost_equal(dot1, dot2)
