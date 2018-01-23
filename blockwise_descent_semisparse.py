from __future__ import absolute_import, division, print_function

import numpy
from sklearn.base import BaseEstimator

from .utils import S, norm_non0

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL(BaseEstimator):
    """A Semi-Sparse Group Lasso model using a blockwise descent solver

    Implements the methods presented by Noah Simon et. al [1], adding the
    notion of the semi-sparse model, as introduced by Romain Tavenard [2].

    Attributes
    ----------
    ind_sparse : numpy.ndarray
    groups : numpy.ndarray

    alpha : float
        alpha paramater provided to constructor
    lambda_ : float
        lambda_ parameter provided to constructor
    max_iter_outer : int
        max_iter_outer parameter provided to constructor
    max_iter_inner : int
        max_iter_inner parameter provided to constructor
    rtol : float
        rtol parameter provided to constructor
    coef_ : None

    .. [1] Noah Simon, Jerome Friedman, Trevor Hastie, Rob Tibshirani,
    "A Sparse-group Lasso," Journal of Computational and Graphical Statistics,
    Vol 22, Issue 2 (2013),
    <http://www.stanford.edu/~hastie/Papers/SGLpaper.pdf>

    .. [2] Romain Tavenard, Sparse-Group Lasso jupyter notebook, Nov 30, 2016
    <https://github.com/rtavenar/Homework/blob/master/pynb/sparse_group_lasso.ipynb>

    See Also
    --------
    subgradients_semisparse : same model, different solver
    """
    def __init__(self, groups, alpha, lambda_, ind_sparse,
                 max_iter_outer=10000, max_iter_inner=100, rtol=1e-6,
                 warm_start=False):
        """
        Parameters
        ----------
        groups : array-like

        alpha : float

        lambda_ : float

        ind_sparse : array-like

        max_iter_outer : int, optional
            Default: 10000

        max_iter_inner : int, optional
            Default: 100

        rtol : float, optional
            Default: 1e-6

        warm_start : boolean, optional
            If True, use previous value of `coef_` as starting point to fit new data
            Default: False
        """
        self.ind_sparse = numpy.array(ind_sparse)
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.warm_start = warm_start
        self.coef_ = None

    def fit(self, X, y):
        """Fit this SGL model using features X and output y

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : numpy.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.
        """
        # Assumption: group ids are between 0 and max(groups)
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if
        # the dimension should not be pushed
        # towards sparsity and 1 otherwise
        n_groups = numpy.max(self.groups) + 1
        n, d = X.shape
        assert d == self.ind_sparse.shape[0]
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        if not self.warm_start or self.coef_ is None:
            self.coef_ = numpy.random.randn(d)
        t = n / (numpy.linalg.norm(X, 2) ** 2)  # Adaptation of the heuristic (?) from fabianp's code
        for iter_outer in range(self.max_iter_outer):
            beta_old = self.coef_.copy()
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr
                if self.discard_group(X, y, indices_group_k):
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out, perform GD for the group
                    beta_k = self.coef_[indices_group_k]
                    p_l = numpy.sqrt(numpy.sum(indices_group_k))
                    for iter_inner in range(self.max_iter_inner):
                        grad_l = self._grad_l(X, y, indices_group_k)
                        tmp = S(beta_k - t * grad_l, t * alpha_lambda[indices_group_k])
                        tmp *= numpy.maximum(1. - t * (1 - self.alpha) * self.lambda_ * p_l / numpy.linalg.norm(tmp), 0.)
                        if numpy.linalg.norm(tmp - beta_k) / norm_non0(tmp) < self.rtol:
                            self.coef_[indices_group_k] = tmp
                            break
                        beta_k = self.coef_[indices_group_k] = tmp
            if numpy.linalg.norm(beta_old - self.coef_) / norm_non0(self.coef_) < self.rtol:
                break
        return self

    def _grad_l(self, X, y, indices_group, group_zero=False):
        if group_zero:
            beta = self.coef_.copy()
            beta[indices_group] = 0.
        else:
            beta = self.coef_
        n, d = X.shape
        r = y - numpy.dot(X, beta)
        return - numpy.dot(X[:, indices_group].T, r) / n

    @staticmethod
    def _static_grad_l(X, y, indices_group, beta=None):
        n, d = X.shape
        if beta is None:
            beta = numpy.zeros((d, ))
        r = y - numpy.dot(X, beta)
        return - numpy.dot(X[:, indices_group].T, r) / n

    def unregularized_loss(self, X, y):
        """The unregularized loss function (i.e. RSS)

        Returns
        -------
        np.float64
            The unregularized loss
        """
        n, d = X.shape
        return numpy.linalg.norm(y - numpy.dot(X, self.coef_)) ** 2 / (2 * n)

    def loss(self, X, y):
        """Total loss function with regularization

        Returns
        -------
        np.float64
            The regularized loss
        """
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        reg_l1 = numpy.linalg.norm(alpha_lambda * self.coef_, ord=1)
        s = 0
        n_groups = numpy.max(self.groups) + 1
        for gr in range(n_groups):
            indices_group_k = self.groups == gr
            s += numpy.sqrt(numpy.sum(indices_group_k)) * numpy.linalg.norm(self.coef_[indices_group_k])
        reg_l2 = (1. - self.alpha) * self.lambda_ * s
        #print(reg_l1, reg_l2, self.unregularized_loss(X, y))
        return self.unregularized_loss(X, y) + reg_l2 + reg_l1

    def discard_group(self, X, y, ind):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : numpy.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.

        ind : boolean numpy.ndarray
            boolean mask for this groups indices

        Returns
        -------
        boolean
            If true, indicates that the coefficients for this group should
            be discarded.
        """
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        norm_2 = numpy.linalg.norm(S(self._grad_l(X, y, ind, group_zero=True), alpha_lambda[ind]))
        p_l = numpy.sqrt(numpy.sum(ind))
        return norm_2 <= (1 - self.alpha) * self.lambda_ * p_l

    def predict(self, X):
        """Predict response vector using the trained coefficients

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        Returns
        -------
        yhat : numpy.ndarray
            Predicted response vector of length n
        """
        return numpy.dot(X, self.coef_)

    def fit_predict(self, X, y):
        """Fit the model and predict the response vector

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : numpy.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.

        Returns
        -------
        yhat : numpy.ndarray
            Predicted response vector of length n
        """
        return self.fit(X, y).predict(X)

    @classmethod
    def lambda_max(cls, X, y, groups, alpha, ind_sparse=None):
        n, d = X.shape
        n_groups = numpy.max(groups) + 1
        max_min_lambda = -numpy.inf
        if ind_sparse is None:
            ind_sparse = numpy.ones((d, ))
        for gr in range(n_groups):
            indices_group = groups == gr
            sqrt_p_l = numpy.sqrt(numpy.sum(indices_group))
            vec_A = numpy.abs(cls._static_grad_l(X, y, indices_group))
            if alpha > 0.:
                min_lambda = numpy.inf
                breakpoints_lambda = numpy.unique(vec_A / alpha)
                lower = 0.
                for l in breakpoints_lambda:
                    indices_nonzero = vec_A >= alpha * l
                    indices_nonzero_sparse = numpy.logical_and(indices_nonzero, ind_sparse[indices_group] > 0)
                    n_nonzero_sparse = numpy.sum(indices_nonzero_sparse)
                    a = n_nonzero_sparse * alpha ** 2 - (sqrt_p_l * (1. - alpha)) ** 2
                    b = - 2. * alpha * numpy.sum(vec_A[indices_nonzero_sparse])
                    c = numpy.sum(vec_A[indices_nonzero] ** 2)
                    delta = b ** 2 - 4 * a * c
                    if delta >= 0.:
                        candidate0 = (- b - numpy.sqrt(delta)) / (2 * a)
                        candidate1 = (- b + numpy.sqrt(delta)) / (2 * a)
                        if lower <= candidate0 <= l:
                            min_lambda = candidate0
                            break
                        elif lower <= candidate1 <= l:
                            min_lambda = candidate1
                            break
                    lower = l
            else:
                min_lambda = numpy.linalg.norm(numpy.dot(X[:, indices_group].T, y) / n) / sqrt_p_l
            if min_lambda > max_min_lambda:
                max_min_lambda = min_lambda
        return max_min_lambda

    @classmethod
    def candidate_lambdas(cls, X, y, groups, alpha, ind_sparse=None, n_lambdas=5, lambda_min_ratio=.1):
        l_max = cls.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
        return numpy.logspace(numpy.log10(lambda_min_ratio * l_max), numpy.log10(l_max), num=n_lambdas)


class SGL_LogisticRegression(SGL):
    # Up to now, we assume that y is 0 or 1
    def unregularized_loss(self, X, y):  # = -1/n * log-likelihood
        n, d = X.shape
        x_beta = numpy.dot(X, self.coef_)
        y_x_beta = x_beta * y
        log_1_e_xb = numpy.log(1. + numpy.exp(x_beta))
        return numpy.sum(log_1_e_xb - y_x_beta, axis=0) / n

    def _grad_l(self, X, y, indices_group, group_zero=False, beta_zero=False):
        if beta_zero:
            beta = numpy.zeros(self.coef_.shape)
        elif group_zero:
            beta = self.coef_.copy()
            beta[indices_group] = 0.
        else:
            beta = self.coef_
        n, d = X.shape
        exp_xb = numpy.exp(numpy.dot(X, beta))
        ratio = exp_xb / (1. + exp_xb)
        return numpy.sum(X[:, indices_group] * (ratio - y).reshape((n, 1)), axis=0) / n

    @staticmethod
    def _static_grad_l(X, y, indices_group, beta=None):
        n, d = X.shape
        if beta is None:
            ratio = .5
        else:
            exp_xb = numpy.exp(numpy.dot(X, beta))
            ratio = exp_xb / (1. + exp_xb)
        return numpy.sum(X[:, indices_group] * (ratio - y).reshape((n, 1)), axis=0) / n

    def predict(self, X):
        y = numpy.ones((X.shape[0]))
        y[numpy.exp(numpy.dot(X, self.coef_)) < 1.] = 0.
        return y

    @staticmethod
    def __logistic(X, beta):
        return 1. / (1. + numpy.exp(numpy.dot(X, beta)))


if __name__ == "__main__":
    n = 1000
    d = 20
    groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
    alpha = .5
    epsilon = .001

    numpy.random.seed(0)
    X = numpy.random.randn(n, d)
    secret_beta = numpy.random.randn(d)
    ind_sparse = numpy.zeros((d, ))
    for i in range(d):
        if groups[i] == 0 or i % 2 == 0:
            secret_beta[i] = 0
        if i % 2 != 0:
            ind_sparse[i] = 1

    y = numpy.dot(X, secret_beta)

    lambda_max = SGL.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
    print(lambda_max)
    for l in [lambda_max - epsilon, lambda_max + epsilon]:
        model = SGL(groups=groups, alpha=alpha, lambda_=l, ind_sparse=ind_sparse)
        model.fit(X, y)
        print(l, model.coef_)

    print(SGL.candidate_lambdas(X, y, groups=groups, alpha=alpha))
