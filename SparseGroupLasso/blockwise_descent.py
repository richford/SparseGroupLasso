from __future__ import absolute_import, division, print_function

import numpy
from .blockwise_descent_semisparse import SSGL
from .blockwise_descent_semisparse import SSGL_LogisticRegression

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SGL(SSGL):
    """A purely sparse group lasso model using a blockwise descent solver

    Inherits from the semi-sparse version and converts to a fully sparse model
    by setting `self.ind_sparse` to all ones. The constructor is the same as
    the semi-sparse version except for the removal of the ind_sparse parameter.
    For all further details, see the docstring for semi-sparse version,
    blockwise_descent_semisparse.SGL

    See Also
    --------
    blockwise_descent_semisparse.SGL : The semi-sparse version
    """
    def __init__(self, groups, alpha, lambda_,
                 max_iter_outer=10000,
                 max_iter_inner=100, rtol=1e-6,
                 warm_start=False):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.warm_start = warm_start
        self.coef_ = None


class SGL_LogisticRegression(SSGL_LogisticRegression):
    """A purely sparse group lasso model using a blockwise descent solver

    Inherits from the semi-sparse version and converts to a fully sparse model
    by setting `self.ind_sparse` to all ones. The constructor is the same as
    the semi-sparse version except for the removal of the ind_sparse parameter.
    For all further details, see the docstring for semi-sparse version,
    blockwise_descent_semisparse.SGL

    See Also
    --------
    blockwise_descent_semisparse.SGL_LogisticRegression : The semi-sparse version
    """
    def __init__(self, groups, alpha, lambda_, max_iter_outer=10000, max_iter_inner=100, rtol=1e-6,
                 warm_start=False):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.warm_start = warm_start
        self.coef_ = None
