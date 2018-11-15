
import matplotlib.pyplot as plt
import numpy as np

from answers import lml
from answers import grad_lml
from prev_answers import poly_func
from prev_answers import trig_func

# use the dataset from the previous cw
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def gradient_descent(step, step_size, Phi):

    # declare empty lists for steps and function values
    alphas, betas = [], []

    # gradient descent algorithm 
    for i in range(10000):

        alpha, beta = step
        alphas.append(alpha)
        betas.append(beta)

        # maximizing gradient descent algorithm
        step = step + step_size*grad_lml(alpha, beta, Phi, Y)

    # return steps and function values for plotting
    return np.column_stack((alphas, betas)).T

def plot(maximums, orders):
    
    plt.plot(orders, maximums)

    plt.xlabel('$order$')
    plt.ylabel('$max-lml$')

    plt.show()

if __name__ == '__main__':

    maximas = []
    maximums = []
    maximum_order = 11
    orders = range(maximum_order+1)
    for order in orders:
        Phi = trig_func(order, X) # init feature matrix
        path = gradient_descent(np.array([.35, .35]), 1e-5, Phi)
        maxima = (path[0, -1], path[1, -1])
        maximum = lml(maxima[0], maxima[1], Phi, Y)
        maximas.append(maxima)
        maximums.append(maximum)

    assert len(maximas) == maximum_order+1
    assert len(maximums) == maximum_order+1

    # plot for (c), a graph of max-lml vs order
    plot(maximums, orders)

