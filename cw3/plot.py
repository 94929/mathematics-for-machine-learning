
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

from answers import lml
from answers import grad_lml

# use the dataset from the previous cw
N = 25
X = Phi = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def gradient_descent(step, step_size):

    # declare empty lists for steps and function values
    alphas, betas = [], []

    # gradient descent algorithm 
    for i in range(100):

        alpha, beta = step
        alphas.append(alpha)
        betas.append(beta)

        # maximizing gradient descent algorithm
        step = step + step_size*grad_lml(alpha, beta, Phi, Y)

    # return steps and function values for plotting
    return np.column_stack((alphas, betas)).T

def plot_contour_2d(f, path, maxima):

    # preparing the configuration values
    xmin, xmax, xstep = 0, 3, .1
    ymin, ymax, ystep = 0, 3, .1
    x, y = np.meshgrid(
                np.arange(xmin, xmax + xstep, xstep), 
                np.arange(ymin, ymax + ystep, ystep)
            )

    z = f(x, y, Phi, Y)
    _maxima = maxima.reshape(-1, 1)

    # plotting methods
    fig, ax = plt.subplots(figsize=(10, 6))

    lvls = np.logspace(-1, 3, 250)

    ax.contour(x, y, z, levels=lvls, norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(path[0,:-1], path[1,:-1], 
            path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], 
            scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*_maxima, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    plt.show()

if __name__ == '__main__':
    # obtain steps (i.e. path) by running the gradient descent algorithm
    path = gradient_descent(step=np.array([1., 1.]), step_size=.1)
    print('Path a:', path[0,:])
    print('Path b:', path[1,:])

    # set maxima of the given function (should be obtained prior to plot)
    maxima = np.array([2.21670974, 0.39935876])

    # plot contour (2d)
    plot_contour_2d(lml, path, maxima)

