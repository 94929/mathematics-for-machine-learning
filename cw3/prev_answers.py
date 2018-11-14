
""" 
answers.py for previous courseworks
"""

import numpy as np

def poly_func(order, X):
    a = np.ones((X.shape[0], order+1))
    for i in range(1, order+1):
        a[:, i] = a[:, i-1] * X.ravel()
    return a

def trig_func(order, X):
    res = np.ones((X.shape[0], 2 * order + 1))
    for i in range(1, order + 1):
        res[:, 2 * i - 1] = np.sin(2 * np.pi * i * X.ravel())
        res[:, 2 * i] = np.cos(2 * np.pi * i * X.ravel())
    return res
