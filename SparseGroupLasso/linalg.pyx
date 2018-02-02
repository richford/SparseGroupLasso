import cython
from cython.parallel import prange
from libc.math cimport sqrt

import numpy as np
from scipy.linalg.blas import ddot
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef inline cnp.double_t _vec_norm(double[:] x, int shape) nogil:
    cdef double ans = 0
    cdef int i
    for i in range(shape):
        ans = ans + x[i] * x[i]
    return sqrt(ans)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef vec_norm(x):
    """Fast vector norm computation for the simple case of a 1D double array """
    return _vec_norm(x, x.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cdef inline cnp.ndarray[double, ndim=1] _dot(
                                        double[:, :] X,
                                        double[:] beta,
                                        int X_shape_0,
                                        int beta_shape):
    out = np.zeros(X_shape_0)

    cdef int i
    cdef int j

    for i in range(X_shape_0):
        for j in range(beta_shape):
            out[i] = out[i] + X[i, j] * beta[j]

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cpdef dot(double[:, :]X, double[:] beta):
    """ Not actually a faster version of a dot product """
    return _dot(X, beta, X.shape[0], beta.shape[0])
