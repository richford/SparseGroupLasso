import numpy
from SparseGroupLasso.utils import S, norm_non0, discard_group

class SSGL_subgrad:
    def __init__(self, groups, alpha, lambda_, ind_sparse, max_iter=10000,
                 rtol=1e-6):
        self.ind_sparse = numpy.array(ind_sparse)
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.rtol = rtol
        self.coef_ = None

    def fit(self, X, y):
        # Assumption: group ids are between 0 and max(groups)
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if the dimension should not be pushed
        # towards sparsity and 1 otherwise
        n_groups = numpy.max(self.groups) + 1
        n, d = X.shape
        assert d == self.ind_sparse.shape[0]
        alpha_lambda = self.alpha * self.lambda_ * self.ind_sparse
        self.coef_ = numpy.random.randn(d)
        for iter in range(self.max_iter):
            beta_old = self.coef_.copy()
            for gr in range(n_groups):
                # 1- Should the group be zero-ed out?
                indices_group_k = self.groups == gr
                if discard_group(X, y, self.coef_, self.alpha, self.lambda_, alpha_lambda, indices_group_k):
                    self.coef_[indices_group_k] = 0.
                else:
                    # 2- If the group is not zero-ed out, update each component
                    p_l = numpy.sqrt(numpy.sum(indices_group_k))
                    for i in range(d):
                        if self.groups[i] == gr:
                            norm2_beta_k = norm_non0(self.coef_[indices_group_k])
                            X_i_k = X[:, i]
                            r_no_i = y - numpy.dot(X, self.coef_) + self.coef_[i] * X_i_k
                            denom = numpy.dot(X_i_k.T, X_i_k) / n + (1 - self.alpha) * self.lambda_ * p_l / norm2_beta_k
                            self.coef_[i] = S(numpy.dot(X_i_k.T, r_no_i) / n, alpha_lambda[i]) / denom
            norm_beta = norm_non0(self.coef_)
            if numpy.linalg.norm(self.coef_ - beta_old) / norm_beta < self.rtol:
                break
        return self

    def predict(self, X):
        return numpy.dot(X, self.coef_)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
