import numpy
from SparseGroupLasso import SSGL_LogisticRegression


def test_sgl_logistic():
    n = 1000
    d = 20
    groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
    alpha = 0.
    lbda = .01

    X = numpy.random.randn(n, d)
    secret_beta = numpy.random.randn(d)
    ind_sparse = numpy.zeros((d, ))
    for i in range(d):
        if groups[i] == 0 or i % 2 == 0:
            secret_beta[i] = 0
        if i % 2 != 0:
            ind_sparse[i] = 1

    y = numpy.ones((n, ))
    y[numpy.exp(numpy.dot(X, secret_beta)) < .5] = 0

    #model = subgradients.SGL(groups=groups, alpha=0., lambda_=0.1)
    #model = subgradients_semisparse.SGL(groups=groups, alpha=0.1, lambda_=0.1, ind_sparse=ind_sparse)
    #model = blockwise_descent.SGL(groups=groups, alpha=0., lambda_=0.1)
    model = SSGL_LogisticRegression(groups=groups, alpha=alpha, lambda_=lbda,
                                    ind_sparse=ind_sparse, max_iter_outer=500)

    model.fit(X, y)
    beta_hat = model.coef_

    err = numpy.linalg.norm(secret_beta - beta_hat)
    epsilon = lbda * 100
    assert err < epsilon
