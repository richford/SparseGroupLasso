import numpy
from SparseGroupLasso import SSGL, SGL, SGL_subgrad, SSGL_subgrad

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_SGL():
    n = 1000
    d = 20
    groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
    alpha = 0.
    for lbda in[0.01, 0.001, 0.0001]:

        X = numpy.random.randn(n, d)
        secret_beta = numpy.random.randn(d)
        ind_sparse = numpy.zeros((d, ))
        for i in range(d):
            if groups[i] == 0 or i % 2 == 0:
                secret_beta[i] = 0
            if i % 2 != 0:
                ind_sparse[i] = 1

        y = numpy.dot(X, secret_beta)

        for model in [SGL_subgrad(groups=groups, alpha=alpha, lambda_=lbda),
                      SSGL_subgrad(groups=groups, alpha=alpha, lambda_=lbda,
                                   ind_sparse=ind_sparse),
                      SGL(groups=groups, alpha=alpha, lambda_=lbda),
                      SSGL(groups=groups, alpha=alpha, lambda_=lbda,
                           ind_sparse=ind_sparse)]:

            model.fit(X, y)
            beta_hat = model.coef_

            err = numpy.linalg.norm(secret_beta - beta_hat)
            epsilon = lbda * 10
            assert err < epsilon
