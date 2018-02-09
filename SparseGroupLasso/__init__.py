from __future__ import absolute_import, division, print_function

from .blockwise_descent import SGL, SGL_LogisticRegression
from .blockwise_descent_semisparse import SSGL, SSGL_LogisticRegression
from .subgradients import SGL_subgrad
from .subgradients_semisparse import SSGL_subgrad
from . import linalg

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
