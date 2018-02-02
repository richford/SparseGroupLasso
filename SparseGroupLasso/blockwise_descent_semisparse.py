from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.base import BaseEstimator
from numba import jit

from .utils import S, norm_non0

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SSGL(BaseEstimator):
    """A Semi-Sparse Group Lasso model using a blockwise descent solver.

    Implements the methods presented by Noah Simon et. al [1], adding the
    notion of the semi-sparse model, as introduced by Romain Tavenard [2].

    Attributes
    ----------
    ind_sparse : np.ndarray
    groups : np.ndarray

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
        self.ind_sparse = np.array(ind_sparse)
        self.groups = np.array(groups)
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
        X : np.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : np.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.
        """
        # Assumption: group ids are between 0 and max(groups)
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if
        # the dimension should not be pushed
        # towards sparsity and 1 otherwise
        n_groups = np.max(self.groups) + 1
        n, d = X.shape
        assert d == self.ind_sparse.shape[0]
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.random.randn(d)

        # Adaptation of the heuristic (?) from fabianp's code:
        t = n / (np.linalg.norm(X, 2) ** 2)
        for iter_outer in range(self.max_iter_outer):
            beta_old = self.coef_.copy()
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr
                X_group_t = X[:, indices_group_k].T
                grad_l = self._grad_l(X, X_group_t, y, indices_group_k,
                                      group_zero=True)
                if self.discard_group(grad_l, indices_group_k):
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out,
                    # perform GD for the group:
                    beta_k = self.coef_[indices_group_k]
                    p_l = np.sqrt(np.sum(indices_group_k))
                    for iter_inner in range(self.max_iter_inner):
                        grad_l = self._grad_l(
                            X, X_group_t, y, indices_group_k)
                        tmp = S(beta_k - t * grad_l, t *
                                alpha_lambda[indices_group_k])
                        norm_tmp = np.sqrt(np.dot(tmp, tmp))
                        # Equation 12 in Simon paper:
                        step = (1. -
                                (t * (1 - self.alpha) * self.lambda_ * p_l /
                                 norm_tmp))
                        tmp *= np.maximum(step, 0.)
                        tmp_beta_k = tmp - beta_k
                        norm_tmp_beta_k = np.sqrt(np.dot(tmp_beta_k,
                                                         tmp_beta_k))
                        norm_non0_tmp = norm_non0(tmp)
                        if norm_tmp_beta_k / norm_non0_tmp < self.rtol:
                            self.coef_[indices_group_k] = tmp
                            break
                        beta_k = self.coef_[indices_group_k] = tmp
            beta_old_coef = beta_old - self.coef_
            if ((np.sqrt(np.dot(beta_old_coef, beta_old_coef)) /
                 norm_non0(self.coef_)) < self.rtol):
                break
        return self

    #@jit
    def _grad_l(self, X, X_group_t, y, indices_group, group_zero=False):
        if group_zero:
            beta = self.coef_.copy()
            beta[indices_group] = 0.
        else:
            beta = self.coef_
        n, d = X.shape
        r = y - np.dot(X, beta)
        return -np.dot(X_group_t, r) / n

    @staticmethod
    def _static_grad_l(X, X_group_t, y, indices_group, beta=None):
        n, d = X.shape
        if beta is None:
            beta = np.zeros((d, ))
        r = y - np.dot(X, beta)
        return -np.dot(X_group_t, r) / n

    def unregularized_loss(self, X, y):
        """The unregularized loss function (i.e. RSS)

        Returns
        -------
        np.float64
            The unregularized loss
        """
        n, d = X.shape
        r = y - np.dot(X, self.coef_)
        return np.dot(r, r) / (2 * n)

    def loss(self, X, y):
        """Total loss function with regularization

        Returns
        -------
        np.float64
            The regularized loss
        """
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        reg_l1 = np.linalg.norm(alpha_lambda * self.coef_, ord=1)
        s = 0
        n_groups = np.max(self.groups) + 1
        for gr in range(n_groups):
            indices_group_k = self.groups == gr
            s += np.sqrt(np.sum(indices_group_k)) * np.sqrt(np.dot(self.coef_[indices_group_k],
                           self.coef_[indices_group_k]))
        reg_l2 = (1. - self.alpha) * self.lambda_ * s
        #print(reg_l1, reg_l2, self.unregularized_loss(X, y))
        return self.unregularized_loss(X, y) + reg_l2 + reg_l1


    def discard_group(self, grad_l, ind):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : np.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.

        ind : boolean np.ndarray
            boolean mask for this groups indices

        Returns
        -------
        boolean
            If true, indicates that the coefficients for this group should
            be discarded.
        """
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        this_S = S(grad_l, alpha_lambda[ind])
        norm_2 = np.sqrt(np.dot(this_S, this_S))
        p_l = np.sqrt(np.sum(ind))
        return norm_2 <= (1 - self.alpha) * self.lambda_ * p_l

    def predict(self, X):
        """Predict response vector using the trained coefficients

        Parameters
        ----------
        X : np.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        Returns
        -------
        yhat : np.ndarray
            Predicted response vector of length n
        """
        return np.dot(X, self.coef_)

    def fit_predict(self, X, y):
        """Fit the model and predict the response vector

        Parameters
        ----------
        X : np.ndarray
            Feature matrix used to train this SGL model. Dimensions are n x p,
            where n is the number of samples and p is the number of features

        y : np.ndarray
            Response vector used to train this SGL model. Length is n,
            where n is the number of samples.

        Returns
        -------
        yhat : np.ndarray
            Predicted response vector of length n
        """
        return self.fit(X, y).predict(X)

    @classmethod
    def lambda_max(cls, X, y, groups, alpha, ind_sparse=None):
        n, d = X.shape
        n_groups = np.max(groups) + 1
        max_min_lambda = -np.inf
        if ind_sparse is None:
            ind_sparse = np.ones((d, ))
        for gr in range(n_groups):
            indices_group = groups == gr
            sqrt_p_l = np.sqrt(np.sum(indices_group))
            X_group_t = X[:, indices_group].T
            vec_A = np.abs(cls._static_grad_l(X, X_group_t, y, indices_group))
            if alpha > 0.:
                min_lambda = np.inf
                breakpoints_lambda = np.unique(vec_A / alpha)
                lower = 0.
                for l in breakpoints_lambda:
                    indices_nonzero = vec_A >= alpha * l
                    indices_nonzero_sparse = np.logical_and(indices_nonzero, ind_sparse[indices_group] > 0)
                    n_nonzero_sparse = np.sum(indices_nonzero_sparse)
                    a = n_nonzero_sparse * alpha ** 2 - (sqrt_p_l * (1. - alpha)) ** 2
                    b = - 2. * alpha * np.sum(vec_A[indices_nonzero_sparse])
                    c = np.sum(vec_A[indices_nonzero] ** 2)
                    delta = b ** 2 - 4 * a * c
                    if delta >= 0.:
                        candidate0 = (- b - np.sqrt(delta)) / (2 * a)
                        candidate1 = (- b + np.sqrt(delta)) / (2 * a)
                        if lower <= candidate0 <= l:
                            min_lambda = candidate0
                            break
                        elif lower <= candidate1 <= l:
                            min_lambda = candidate1
                            break
                    lower = l
            else:
                min_lambda = np.linalg.norm(np.dot(X[:, indices_group].T, y) / n) / sqrt_p_l
            if min_lambda > max_min_lambda:
                max_min_lambda = min_lambda
        return max_min_lambda

    @classmethod
    def candidate_lambdas(cls, X, y, groups, alpha, ind_sparse=None, n_lambdas=5, lambda_min_ratio=.1):
        l_max = cls.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
        return np.logspace(np.log10(lambda_min_ratio * l_max), np.log10(l_max), num=n_lambdas)


class SSGL_LogisticRegression(SSGL):
    # Up to now, we assume that y is 0 or 1
    def unregularized_loss(self, X, y):  # = -1/n * log-likelihood
        n, d = X.shape
        x_beta = np.dot(X, self.coef_)
        y_x_beta = x_beta * y
        log_1_e_xb = np.log(1. + np.exp(x_beta))
        return np.sum(log_1_e_xb - y_x_beta, axis=0) / n

    def _grad_l(self, X, X_group_t, y, indices_group, group_zero=False,
                beta_zero=False):
        if beta_zero:
            beta = np.zeros(self.coef_.shape)
        elif group_zero:
            beta = self.coef_.copy()
            beta[indices_group] = 0.
        else:
            beta = self.coef_
        n, d = X.shape
        exp_xb = np.exp(np.dot(X, beta))
        ratio = exp_xb / (1. + exp_xb)
        return np.sum(X_group_t.T * (ratio - y).reshape((n, 1)), axis=0) / n

    @staticmethod
    def _static_grad_l(X, X_group_t, y, indices_group, beta=None):
        n, d = X.shape
        if beta is None:
            ratio = .5
        else:
            exp_xb = np.exp(np.dot(X, beta))
            ratio = exp_xb / (1. + exp_xb)
        return np.sum(X_group_t.T * (ratio - y).reshape((n, 1)),
                      axis=0) / n

    def predict(self, X):
        y = np.ones((X.shape[0]))
        y[np.exp(np.dot(X, self.coef_)) < 1.] = 0.
        return y

    @staticmethod
    def __logistic(X, beta):
        return 1. / (1. + np.exp(np.dot(X, beta)))


if __name__ == "__main__":
    n = 1000
    d = 20
    groups = np.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
    alpha = .5
    epsilon = .001

    np.random.seed(0)
    X = np.random.randn(n, d)
    secret_beta = np.random.randn(d)
    ind_sparse = np.zeros((d, ))
    for i in range(d):
        if groups[i] == 0 or i % 2 == 0:
            secret_beta[i] = 0
        if i % 2 != 0:
            ind_sparse[i] = 1

    y = np.dot(X, secret_beta)

    lambda_max = SGL.lambda_max(X, y, groups=groups, alpha=alpha, ind_sparse=ind_sparse)
    print(lambda_max)
    for l in [lambda_max - epsilon, lambda_max + epsilon]:
        model = SGL(groups=groups, alpha=alpha, lambda_=l, ind_sparse=ind_sparse)
        model.fit(X, y)
        print(l, model.coef_)

    print(SGL.candidate_lambdas(X, y, groups=groups, alpha=alpha))
