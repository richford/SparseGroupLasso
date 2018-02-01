"""
To use in IPython

%load_ext line_profiler
import profile_sgl as p
from SparseGroupLasso import SGL
%lprun -f SGL.fit p.func()

"""
import numpy as np
from SparseGroupLasso import SSGL, SGL, SGL_subgrad, SSGL_subgrad

n = 1000
d = 20
groups = np.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
alpha = 0.
for lbda in[0.01, 0.001, 0.0001]:

    X = np.random.randn(n, d)
    secret_beta = np.random.randn(d)
    ind_sparse = np.zeros((d, ))
    for i in range(d):
        if groups[i] == 0 or i % 2 == 0:
            secret_beta[i] = 0
        if i % 2 != 0:
            ind_sparse[i] = 1

    y = np.dot(X, secret_beta)

models = {'SGL_subgrad': SGL_subgrad(groups=groups, alpha=alpha, lambda_=lbda),
          'SSGL_subgrad': SSGL_subgrad(groups=groups, alpha=alpha,
                                       lambda_=lbda, ind_sparse=ind_sparse),
          'SGL': SGL(groups=groups, alpha=alpha, lambda_=lbda),
          'SSGL': SSGL(groups=groups, alpha=alpha, lambda_=lbda,
                       ind_sparse=ind_sparse)}

model = models['SGL']


def func():
    model.fit(X, y)


if __name__=="__main__":
    func()
