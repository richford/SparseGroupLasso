import numpy
from SparseGroupLasso.subgradients_semisparse import SSGL_subgrad

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL_subgrad(SSGL_subgrad):
    """A purely sparse group lasso model using a subgradient solver

    Inherits from the semi-sparse version and converts to a fully sparse model
    by setting `self.ind_sparse` to all ones. The constructor is the same as
    the semi-sparse version except for the removal of the ind_sparse parameter.
    For all further details, see the docstring for semi-sparse version,
    subgradients_semisparse.SGL

    See Also
    --------
    subgradients_semisparse.SGL : The semi-sparse version
    """
    def __init__(self, groups, alpha, lambda_, max_iter=1000, rtol=1e-6):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.rtol = rtol
        self.coef_ = None
