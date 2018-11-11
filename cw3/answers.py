# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

import numpy as np

def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """

    N, M = Phi.shape
    bI = beta * np.eye(N)

    term = alpha*Phi.dot(Phi.T) + bI
    term_1 = -0.5 * N * np.log(2*np.pi)
    term_2 = -0.5 * np.log(np.linalg.det(term))
    term_3 = -0.5 * Y.T.dot(np.linalg.inv(term)).dot(Y)

    return np.asscalar(term_1 + term_2 + term_3)

def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
    """
    pass

